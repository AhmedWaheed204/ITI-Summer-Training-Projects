import cv2
import numpy as np
import streamlit as st
import os
from datetime import datetime, timedelta

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load face detection model
face_cascade = cv2.CascadeClassifier(r'D:\Ahmed\ITI\Projects\Pro6\Pro6c\haarcascade_frontalface_default.xml')

# Load known faces
known_faces = {}
known_faces_dir = os.path.join(current_dir, 'known_faces')

if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)
    st.warning(f"The 'known_faces' directory was created at {known_faces_dir}. Please add some face images to this directory and restart the app.")

for filename in os.listdir(known_faces_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(known_faces_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        known_faces[name] = image

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces, gray

def simple_face_recognition(face_roi):
    if not known_faces:
        return "Unknown"
    
    best_match = None
    best_score = float('inf')
    
    for name, known_face in known_faces.items():
        known_face_resized = cv2.resize(known_face, (face_roi.shape[1], face_roi.shape[0]))
        diff = cv2.absdiff(face_roi, known_face_resized)
        score = np.sum(diff)
        
        if score < best_score:
            best_score = score
            best_match = name
    
    # You may need to adjust this threshold based on your specific use case
    if best_score < 5000000:  # This is an arbitrary threshold, adjust as needed
        return best_match
    else:
        return "Unknown"

def main():
    st.title("Face Recognition Unlock System")

    if not known_faces:
        st.error("No known faces found. Please add some face images to the 'known_faces' directory and restart the app.")
        return

    # Load the specific image
    image_path = r"D:\Ahmed\ITI\Projects\Pro6\Pro6d\known_faces\Hillary.jpg"
    frame = cv2.imread(image_path)
    
    if frame is None:
        st.error("Failed to load the image. Please check the file path.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.write("Input Image")
        st.image(frame, channels="BGR", use_column_width=True)

    with col2:
        st.write("System Status")
        status_placeholder = st.empty()
        unlock_button = st.button("Attempt Unlock")

    faces, gray = detect_faces(frame)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        name = simple_face_recognition(face_roi)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.write("Processed Image")
    st.image(frame, channels="BGR", use_column_width=True)

    if unlock_button:
        if len(faces) > 0 and name != "Unknown":
            status_placeholder.success(f"Access Granted! Welcome, {name}")
        else:
            status_placeholder.error("Access Denied. Face not recognized or no face detected.")
    else:
        status_placeholder.info("System Locked. Click 'Attempt Unlock' to try unlocking.")

if __name__ == "__main__":
    main()
