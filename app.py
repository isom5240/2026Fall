
# import part
import streamlit as st
from transformers import pipeline

# function part


# main part
# Set up page configuration
st.set_page_config(
    page_title="ISOM5240: Sentiment Analysis App",
    page_icon="😊",
    layout="centered"
)

# Title and description
st.title("😊 Sentiment Analysis App")
st.write("Analyze the sentiment of your text using Hugging Face Transformers.")

# Text input area
default_text = "Deep Learning (DL) represents a highly promising approach to developing applications in Artificial Intelligence (AI)."
user_input = st.text_area("Enter text to analyze:", value=default_text, height=150)

# Analyze button
if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            # Load model directly without caching
            sentiment_pipeline = pipeline("sentiment-analysis")
            result = sentiment_pipeline(user_input)[0]
            
            label = result["label"]
            score = result["score"]

        st.subheader("Result")
        
        # Display formatted output based on sentiment
        if label.upper() == "POSITIVE":
            st.success(f"**Sentiment:** {label} 🎉")
        else:
            st.error(f"**Sentiment:** {label} 🙁")
            
        st.metric(label="Confidence Score", value=f"{score:.4f}")
      
