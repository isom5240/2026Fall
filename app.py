import streamlit as st
from transformers import pipeline
from PIL import Image

# Load the age classification pipeline directly on script execution
age_classifier = pipeline("image-classification",
                          model="nateraw/vit-age-classifier")


def classify_age(image):
    """Classify the age of a person in the given image."""
    results = age_classifier(image)

    # Sort results by score in descending order
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    return results


# Streamlit UI
st.title("Age Classification using ViT")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Fixed: use_container_width replaces the deprecated use_column_width
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Classify age
    st.text('Classifying age...')
    age_predictions = classify_age(image)

    # Display results
    st.subheader("Predicted Age Range:")
    st.write(f"Age range: {age_predictions[0]['label']}")
