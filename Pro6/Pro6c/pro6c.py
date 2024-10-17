import cv2
import numpy as np
import streamlit as st
import os

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

def detect_face_and_eyes(image):
    # Load pre-trained models
    face_cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
    eye_cascade_path = os.path.join(current_dir, 'haarcascade_eye.xml')
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty():
        st.error(f"Error: Unable to load face cascade classifier. Check if the file exists at {face_cascade_path}")
        return image

    if eye_cascade.empty():
        st.error(f"Error: Unable to load eye cascade classifier. Check if the file exists at {eye_cascade_path}")
        return image

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    return image

# Streamlit app
def main():
    st.title("Face and Eye Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        
        # Perform detection
        result = detect_face_and_eyes(image)

        # Display the result
        st.image(result, channels="BGR", caption="Detected Faces and Eyes")

if __name__ == "__main__":
    main()
