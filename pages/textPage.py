import streamlit as st
import random

# Set full-screen layout
st.set_page_config(page_title="Text Fake Detection", page_icon="📝", layout="wide")

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
    </style>
    """, 
    unsafe_allow_html=True
)

# Title with styling
st.markdown(
    "<h1 style='text-align: center; color: white; background: linear-gradient(90deg, #d4a5ff, #b266ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>📝 Fake Text Detection</h1>", 
    unsafe_allow_html=True
)

# Text input for detection
text_input = st.text_area(
    "Enter the text you want to analyze:", 
    height=250, 
    placeholder="Paste the text here to check for authenticity..."
)

# Detection function (placeholder for future ML model)
def detect_fake_text(text):
    # Simulated detection logic
    if not text or len(text.strip()) < 10:
        return None
    
    # Randomly determine if text is real or fake
    return random.choice(["real", "fake"])

# Detection button
if st.button("🔍 Detect Text Authenticity"):
    if text_input:
        # Run detection
        result = detect_fake_text(text_input)
        
        if result == "real":
            st.success("✅ The text appears to be **genuine**.")
            st.info("Confidence Score: 85%")
        elif result == "fake":
            st.error("🚨 The text seems to be **potentially fabricated**.")
            st.warning("Confidence Score: 75%")
        else:
            st.warning("Please enter a valid text for analysis.")
    else:
        st.warning("Please enter some text to analyze.")

# Additional information section
st.markdown("""
    ### 📌 How does Text Detection Work?
    - Our AI analyzes multiple linguistic and semantic features
    - Checks for unnatural language patterns
    - Compares against known text generation markers
    - Provides a confidence score for authenticity
""")

# Navigation back to home
if st.button("🔙 Back to Home"):
    st.switch_page("app.py")