import streamlit as st
from streamlit_webrtc import webrtc_streamer
webrtc_streamer(k="emotion_stream")
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import os
import webbrowser
import tensorflow as tf

# Paths to model and label files
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.npy"

# Load model and labels
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(LABELS_PATH):
    st.error(f"Labels file not found: {LABELS_PATH}")
    st.stop()

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH)

# Initialize MediaPipe components
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion-Based Music Recommender")

# Session state for emotion detection
if "run" not in st.session_state:
    st.session_state["run"] = True

emotion = ""
if os.path.exists("emotion.npy"):
    emotion = np.load("emotion.npy")[0]

class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

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
            pred = labels[np.argmax(model.predict(lst))]
            cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            np.save("emotion.npy", np.array([pred]))

        # Draw landmarks
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# User inputs
lang = st.text_input("Preferred Language")
singer = st.text_input("Preferred Singer")

if lang and singer and st.session_state["run"]:
    webrtc_streamer(key="emotion_stream", video_processor_factory=EmotionProcessor)

btn = st.button("Recommend Me Songs")

if btn:
    if not os.path.exists("emotion.npy") or not emotion:
        st.warning("Please allow the system to capture your emotion first.")
    else:
        search_query = f"{lang} {emotion} song {singer}"
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = False
