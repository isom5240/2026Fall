
# import part
import streamlit as st
from transformers import pipeline

# Specify the model name explicitly
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# function part
def analyze_sentiment(text):
    """Loads the pipeline and predicts sentiment for the provided text."""
    sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
    results = sentiment_pipeline(text)
    return results[0]

def display_results(label, score):
    """Displays the sentiment prediction and confidence score in Streamlit UI."""
    st.subheader("Result")
    
    if label.upper() == "POSITIVE":
        st.success(f"**Sentiment:** {label} 🎉")
    else:
        st.error(f"**Sentiment:** {label} 🙁")
        
    st.metric(label="Confidence Score", value=f"{score:.4f}")

# main part
def main():
    # Set up page configuration
    st.set_page_config(
        page_title="ISOM5240: Sentiment Analysis App",
        page_icon="😊",
        layout="centered"
    )

    # Title and description
    st.title("😊 Hugging Face")
    st.title("😊 Sentiment Analysis App")
    st.write("Analyze the sentiment of your text")
    st.write("using Hugging Face Transformers.")

    # Text input area
    default_text = "Deep Learning (DL) represents a highly promising approach to developing applications in Artificial Intelligence (AI)."
    user_input = st.text_area("Enter text to analyze:", value=default_text, height=350)

    # Analyze button
    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                result = analyze_sentiment(user_input)
                label = result["label"]
                score = result["score"]

            display_results(label, score)

if __name__ == "__main__":
    main()
