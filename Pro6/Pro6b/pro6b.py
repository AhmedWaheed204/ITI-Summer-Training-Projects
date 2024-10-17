import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Load the Haar Cascade classifier
cascade_path = r"D:\Ahmed\ITI\Projects\Pro6\Pro6b\haarcascade_fullbody.xml"
body_cascade = cv2.CascadeClassifier(cascade_path)

def detect_pedestrians(image):
    # Convert PIL Image to OpenCV format
    image = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect pedestrians
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw bounding boxes
    for (x, y, w, h) in bodies:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image, len(bodies)

def main():
    st.title("Pedestrian Detection App")
    
    st.sidebar.header("Detection Settings")
    scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05)
    min_neighbors = st.sidebar.slider("Minimum Neighbors", 1, 10, 5)
    min_size = st.sidebar.slider("Minimum Size", 10, 100, 30)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Pedestrians"):
            result_image, num_pedestrians = detect_pedestrians(image)
            st.image(result_image, caption="Result", use_column_width=True)
            st.success(f"Number of pedestrians detected: {num_pedestrians}")

if __name__ == "__main__":
    main()
