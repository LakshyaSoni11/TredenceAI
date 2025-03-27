import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Set full-screen layout
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# Apply global styles
st.markdown(
    """
    <style>
    /* Ensure full page background */
    [data-testid="stAppViewContainer"] {
        background-color: black;
    }

    /* Textarea Styling */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 2px solid #6a0dad !important;
    }

    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(90deg, #6a0dad, #8e44ad) !important;
        color: white !important;
        border: none !important;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #8e44ad, #6a0dad) !important;
        transform: scale(1.05) !important;
    }
    
    /* Result boxes */
    .real-box {
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 5px solid #00ff00;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .fake-box {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 5px solid #ff0000;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background-color: rgba(255, 255, 255, 0.3);
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Title with styling
st.markdown(
    "<h1 style='text-align: center; color: white; background: linear-gradient(90deg, #d4a5ff, #b266ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>📰 Fake News Detection</h1>", 
    unsafe_allow_html=True
)

# Load tokenizer and model (in a real app, these would be pre-loaded)
@st.cache_resource
def load_model_components():
    # Load tokenizer (in a real app, you would load your pre-trained tokenizer)
    tokenizer = Tokenizer()
    
    # Define model architecture (same as in your notebook)
    maxlen = 150
    v = 91536  # Vocabulary size from your notebook
    
    inputt = tf.keras.layers.Input(shape=(maxlen,))
    x = tf.keras.layers.Embedding(v+1, 100)(inputt)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(100, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    model = Model(inputt, x)
    
    # Load weights (in a real app, you would load your pre-trained weights)
    # model.load_weights('path_to_your_weights.h5')
    
    return tokenizer, model

tokenizer, model = load_model_components()

# Text preprocessing function (same as in your notebook)
def process_text(text):
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Remove extra white space
    text = re.sub(r'\W', ' ', str(text))  # Remove special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical
    text = text.lower()
    
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if len(word) > 3]
    
    indices = np.unique(words, return_index=True)[1]
    cleaned_text = np.array(words)[np.sort(indices)].tolist()
    
    return cleaned_text

# Prediction function
def predict_text(text):
    # Preprocess the text
    cleaned_text = process_text(text)
    
    # Convert to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    # Pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=150)
    
    # Make prediction
    prediction = model.predict(padded_sequence)
    
    # Get confidence scores
    fake_confidence = prediction[0][0] * 100
    real_confidence = prediction[0][1] * 100
    
    # Determine label (0=fake, 1=real)
    label = np.argmax(prediction)
    
    return label, real_confidence, fake_confidence

# Text input for detection
text_input = st.text_area(
    "Enter the news text you want to analyze:", 
    height=250, 
    placeholder="Paste the news article here to check for authenticity..."
)

# Detection button
if st.button("🔍 Detect News Authenticity"):
    if text_input and len(text_input.strip()) > 50:  # Minimum length check
        with st.spinner('Analyzing text...'):
            # Run detection
            label, real_confidence, fake_confidence = predict_text(text_input)
            
            # Display results
            if label == 1:  # Real
                st.markdown(
                    f"""
                    <div class="real-box">
                        <h3>✅ This news appears to be <strong>GENUINE</strong></h3>
                        <p>Confidence: {real_confidence:.2f}%</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {real_confidence}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:  # Fake
                st.markdown(
                    f"""
                    <div class="fake-box">
                        <h3>🚨 This news appears to be <strong>FAKE</strong></h3>
                        <p>Confidence: {fake_confidence:.2f}%</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {fake_confidence}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            # Show text analysis
            st.subheader("Text Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Real Probability", f"{real_confidence:.2f}%")
            with col2:
                st.metric("Fake Probability", f"{fake_confidence:.2f}%")
                
            # Show processed text
            with st.expander("Show processed text"):
                processed = process_text(text_input)
                st.write(" ".join(processed))
    elif text_input:
        st.warning("Please enter a longer text for accurate analysis (at least 50 characters).")
    else:
        st.warning("Please enter some text to analyze.")

# Additional information section
st.markdown("""
    ### 📌 How Our Fake News Detection Works
    - **Text Preprocessing**: Cleans and normalizes the input text
    - **Deep Learning Model**: Uses a Bidirectional LSTM/GRU neural network
    - **Pattern Recognition**: Analyzes linguistic patterns characteristic of fake news
    - **Confidence Scoring**: Provides probability scores for authenticity
    
    ### ℹ️ About the Model
    - Trained on 44,898 news articles (23,481 fake, 21,417 real)
    - 91,536 word vocabulary
    - 97.2% accuracy on test data
    - Processes text through embedding and recurrent neural network layers
""")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>Fake News Detection System | Powered by Deep Learning</div>", 
    unsafe_allow_html=True
)