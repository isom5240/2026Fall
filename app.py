

import streamlit as st
from PIL import Image
from transformers import pipeline

# Recommended efficient model for fast, accurate image captioning
MODEL_NAME = "Salesforce/blip-image-captioning-base"


def load_captioning_pipeline():
    """Loads and initializes the Hugging Face image-to-text pipeline."""
    return pipeline("image-to-text", model=MODEL_NAME)


def generate_description(image, captioner):
    """Generates a brief text description from an input PIL image."""
    result = captioner(image)
    return result[0]["generated_text"]


def main():
    st.set_page_config(
        page_title="Image Description Generator",
        page_icon="🖼️",
        layout="centered",
    )

    st.title("🖼️ Image-to-Text Web App")
    st.write(
        "Upload an image to generate a brief, automated description using Hugging Face Transformers."
    )

    # File uploader widget for images
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # Display uploaded image preview
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Generate Description Button
        if st.button("Describe Image", type="primary"):
            with st.spinner(
                "Loading model and generating description (this may take a few seconds on first run)..."
            ):
                captioner = load_captioning_pipeline()
                description = generate_description(image, captioner)

            st.subheader("Generated Description")
            st.success(description.capitalize())


if __name__ == "__main__":
    main()
    
