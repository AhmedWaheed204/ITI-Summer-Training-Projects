import cv2
import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import io

def tempering_detector(original_image, tampered_image):
    # Convert PIL Images to numpy arrays
    original = np.array(original_image.convert('RGB'))
    tampered = np.array(tampered_image.convert('RGB'))
    
    # Resize images to match dimensions
    height = min(original.shape[0], tampered.shape[0])
    width = min(original.shape[1], tampered.shape[1])
    original = cv2.resize(original, (width, height))
    tampered = cv2.resize(tampered, (width, height))
    
    # Convert to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_RGB2GRAY)
    
    # Calculate structural similarity index
    (score, diff) = ssim(original_gray, tampered_gray, full=True)
    diff = (diff * 255).astype("uint8")
    
    # Threshold the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    # Draw contours on tampered image
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    is_tampered = score < 1.0
    return is_tampered, score, tampered

def main():
    st.title("Image Tampering Detection")
    
    original_image = st.file_uploader("Upload original image", type=["jpg", "jpeg", "png"])
    tampered_image = st.file_uploader("Upload potentially tampered image", type=["jpg", "jpeg", "png"])
    
    if original_image is not None and tampered_image is not None:
        original_pil = Image.open(original_image)
        tampered_pil = Image.open(tampered_image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_pil, caption="Original Image", use_column_width=True)
        with col2:
            st.image(tampered_pil, caption="Potentially Tampered Image", use_column_width=True)
        
        if st.button("Detect Tampering"):
            try:
                is_tampered, similarity_score, result_image = tempering_detector(original_pil, tampered_pil)
                
                st.image(result_image, caption="Result Image", use_column_width=True)
                st.write(f"Similarity Score: {similarity_score:.2f}")
                st.write(f"Image is {'tampered' if is_tampered else 'not tampered'}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

