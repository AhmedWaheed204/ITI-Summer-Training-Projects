import cv2
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile

def car_detection(video_file):
    # Load the pre-trained car detection classifier
    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

    # Start video capture from the uploaded video file
    cap = cv2.VideoCapture(video_file)

    st.title("Car Detection")
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars
        cars = car_cascade.detectMultiScale(gray, 1.1, 3)

        # Draw rectangles around detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Convert frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame in Streamlit
        stframe.image(frame_rgb, channels="RGB")
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()

# Streamlit app
st.title("Upload a Video for Car Detection")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Call the car detection function with the path to the temporary file
    car_detection(temp_file_path)
