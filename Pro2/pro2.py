import streamlit as st
from PIL import Image

def add_watermark(original_image, watermark_image):
    # Open the original image
    base_image = original_image.convert('RGBA')
    
    # Open the watermark image
    watermark = watermark_image.convert('RGBA')
    
    # Resize watermark to a percentage of the base image size
    watermark_size = (base_image.width // 6, base_image.height // 6)
    watermark = watermark.resize(watermark_size, Image.LANCZOS)
    
    # Create a new transparent image the same size as the base
    transparent = Image.new('RGBA', base_image.size, (0,0,0,0))
    
    # Paste the watermark onto the transparent image
    position = (base_image.width - watermark.width, base_image.height - watermark.height)
    transparent.paste(watermark, position, watermark)
    
    # Combine the base image with the watermark
    output = Image.alpha_composite(base_image, transparent)
    
    return output

# Streamlit app
st.title('Watermarking App')

# Upload original image
original_image_file = st.file_uploader("Upload Original Image", type=["png", "jpg", "jpeg"])
# Upload watermark image
watermark_image_file = st.file_uploader("Upload Watermark Image", type=["png", "jpg", "jpeg"])

if original_image_file and watermark_image_file:
    original_image = Image.open(original_image_file)
    watermark_image = Image.open(watermark_image_file)
    
    # Add watermark
    result_image = add_watermark(original_image, watermark_image)
    
    # Display result
    st.image(result_image, caption='Watermarked Image')
    
    # Provide download link
    result_image.save("result.png")
    with open("result.png", "rb") as file:
        btn = st.download_button(
            label="Download Image",
            data=file,
            file_name="result.png",
            mime="image/png"
        )
