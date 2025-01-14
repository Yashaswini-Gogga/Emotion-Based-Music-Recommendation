import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Load the model and compile if needed
try:
    model = load_model("model.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
except Exception as e:
    st.error(f"Error loading or compiling the model: {e}")

# Load labels
try:
    label = np.load("labels.npy")
except Exception as e:
    st.error(f"Error loading labels: {e}")
    label = []

# MediaPipe setup
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

# Initialize session state
if "run" not in st.session_state:
    st.session_state["run"] = "true"

# Load emotion state
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Define EmotionProcessor class
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)  # Flip frame horizontally

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
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)

            try:
                pred = label[np.argmax(model.predict(lst))]
                cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
                np.save("emotion.npy", np.array([pred]))
            except Exception as e:
                st.error(f"Error in prediction: {e}")

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                                connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Input fields for language and singer
lang = st.text_input("Language")
singer = st.text_input("Singer")

# WebRTC configuration
if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(
        key="key",
        desired_playing_state=True,
        video_processor_factory=EmotionProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# Recommend songs button
btn = st.button("Recommend me songs")

if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
