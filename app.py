
import streamlit as st
from transformers import pipeline

# Title and description
st.title("Sentiment Analysis App")
st.write("Enter text below to analyze its sentiment using Hugging Face's pipeline.")

# Cache the model pipeline so it doesn't reload on every interaction
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_sentiment_pipeline()

# User input text area
text_input = st.text_area(
    "Input Text",
    value="Deep Learning (DL) represents a highly promising approach to developing applications in Artificial Intelligence (AI).",
    height=150
)

# Analyze button
if st.button("Analyze Sentiment"):
    if text_input.strip():
        result = sentiment_pipeline(text_input)
        label = result[0]["label"]
        score = result[0]["score"]
        
        # Display results with metrics
        st.subheader("Result")
        col1, col2 = st.columns(2)
        col1.metric("Sentiment", label)
        col2.metric("Confidence Score", f"{score:.4f}")
    else:
        st.warning("Please enter some text to analyze.")
      
