import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Load model and labels with proper error handling
try:
    model = load_model("model.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    label = np.load("labels.npy")
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    model = None

# Initialize mediapipe for holistic and hands
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

# Initialize session state
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Load previous emotion (if any)
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# EmotionProcessor class for handling emotion detection
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        ##############################
        frm = cv2.flip(frm, 1)

        # Process the frame using mediapipe holistic model
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            # Ensure model is loaded before prediction
            if model:
                try:
                    pred = label[np.argmax(model.predict(lst))]
                    np.save("emotion.npy", np.array([pred]))
                    cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
            else:
                cv2.putText(frm, "Model not loaded", (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

        # Draw landmarks on the frame
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        ##############################
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Input fields for language and singer
lang = st.text_input("Language")
singer = st.text_input("Singer")

# Only start streaming when both language and singer are provided, and model has detected emotion
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=EmotionProcessor)

# Button to recommend songs based on the detected emotion
btn = st.button("Recommend me songs")

if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first.")
        st.session_state["run"] = "true"
    else:
        # Construct search query for YouTube
        search_query = f"{lang} {emotion} song {singer}"
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        np.save("emotion.npy", np.array([""]))  # Reset emotion state after recommendation
        st.session_state["run"] = "false"
