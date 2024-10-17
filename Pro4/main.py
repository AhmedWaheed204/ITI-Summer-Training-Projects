import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from Swapping_fun import swap_faces  # Assuming you have a face_swap module with a swap_faces function

def main():
    st.title("Face Swapping App")

    # File uploaders for source and target images
    source_file = st.file_uploader("Choose a source face image", type=["jpg", "jpeg", "png"])
    target_file = st.file_uploader("Choose a target image", type=["jpg", "jpeg", "png"])

    if source_file and target_file:
        source_image = Image.open(source_file)
        target_image = Image.open(target_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(source_image, caption="Source Image", use_column_width=True)
        with col2:
            st.image(target_image, caption="Target Image", use_column_width=True)

        if st.button("Swap Faces"):
            # Convert PIL Images to numpy arrays
            source_array = np.array(source_image)
            target_array = np.array(target_image)

            # Perform face swapping
            result = swap_faces(source_array, target_array)

            # Display the result
            st.image(result, caption="Face Swapped Result", use_column_width=True)

if __name__ == "__main__":
    main()
