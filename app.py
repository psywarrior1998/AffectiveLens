# --- 1. Core Imports for Streamlit and the NLP Pipeline ---
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel
import torch
import os
import gc

# --- 2. Global Configuration and Model Caching ---
# Define repository and model paths from the project's Hugging Face assets.
# This makes the app self-contained and reproducible.
MODEL_REPO_ID = "psyrishi/affectivelens-emotion-models"
MODEL_FILENAME = "LightGBM_MicroF1_0.6240.pkl"  # The champion model identified in the notebook.
MODEL_CHECKPOINT = "distilbert-base-uncased"
BROAD_EMOTION_CATEGORIES = ['negative', 'neutral', 'positive']
positive_indices = {0, 4, 11, 23, 17, 10, 22, 18, 13, 27, 26, 15, 20, 19}
negative_indices = {2, 5, 21, 6, 9, 3, 12, 16, 7, 24, 25, 8, 1, 14}


# A precedence rule: Negative > Positive > Neutral, as defined in the notebook.

# Use Streamlit's caching to load heavy models once.
# This prevents reloading on every user interaction, making the app performant.
@st.cache_resource
def load_models():
    """
    Loads the Hugging Face DistilBERT model and the champion classifier from the Hub.
    The function runs only once for the entire session.
    """
    st.write("Loading Transformer and Classifier models... This may take a moment.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download the champion classifier model from the Hugging Face Hub.
    # The `resume_download` parameter is now removed.
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    classifier_model = joblib.load(model_path)

    # Load the DistilBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    transformer_model = AutoModel.from_pretrained(MODEL_CHECKPOINT)
    transformer_model.eval()
    transformer_model.to(device)

    return tokenizer, transformer_model, classifier_model, device


# Call the caching function to load models at the start of the app
tokenizer, transformer_model, final_model, device = load_models()


# --- 3. The Core Prediction Function (from AffectiveLens.ipynb) ---
def predict_emotion(text: str):
    """
    Predicts a single broad emotion category for a given text using the best trained model.
    This logic is directly from the project's Jupyter notebook.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the embedding from the Transformer model
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        text_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # Use the pre-loaded champion classifier to predict
    predicted_class_index = final_model.predict(text_embedding)[0]
    predicted_emotion_name = BROAD_EMOTION_CATEGORIES[predicted_class_index]

    return predicted_emotion_name


# --- 4. Streamlit Application UI Layout ---
st.title('AffectiveLens: Emotion Detection Engine ✨')
st.markdown(
    'A Transformer-based system for classifying text emotion as positive, negative, or neutral. Powered by DistilBERT and LightGBM.')

# Create a text area for user input
user_input = st.text_area(
    "Enter a sentence or a paragraph below:",
    height=150,
    placeholder="e.g., I am absolutely thrilled with the results, feeling pure joy and excitement!"
)

# Create a predict button
if st.button('Classify Emotion'):
    if user_input:
        # Show a loading spinner while processing
        with st.spinner('Analyzing emotion...'):
            predicted_emotion = predict_emotion(user_input)

        # Display the prediction in a prominent way
        st.success(f"**Predicted Emotion:** **:green[{predicted_emotion.upper()}]**")

        # Add a clear section for more information
        st.markdown(f"### What does this mean?")
        if predicted_emotion == 'positive':
            st.info(
                "The model detected **positive** emotional content. This often includes feelings like joy, excitement, and love.")
        elif predicted_emotion == 'negative':
            st.info(
                "The model detected **negative** emotional content. This often includes feelings like sadness, anger, and disappointment.")
        else:
            st.info(
                "The model detected **neutral** emotional content. The text may be factual or ambiguous, lacking a clear emotional valence.")
    else:
        st.warning('Please enter some text to classify.')

# --- 5. Memory Cleanup ---
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()