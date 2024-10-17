import pytesseract
from PIL import Image
import streamlit as st
import io

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image(image, lang):
    text = pytesseract.image_to_string(image, lang=lang)
    return text

def put_text_in_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def main():
    st.title("OCR - Extract Text from Images (Arabic & English)")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Add language selection
    lang = st.selectbox("Select language", ["eng", "ara", "eng+ara"], 
                        format_func=lambda x: "English" if x == "eng" else "Arabic" if x == "ara" else "English + Arabic")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Extract Text'):
            text = extract_text_from_image(image, lang)
            
            st.subheader("Extracted Text:")
            st.text_area(label="Extracted Text", value=text, height=300)
            
            # Option to download the extracted text as a file
            if text.strip():  # Only show download button if there's text
                st.download_button(
                    label="Download text as file",
                    data=text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No text was extracted from the image.")

if __name__ == "__main__":
    main()
