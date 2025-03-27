import streamlit as st

st.title("🎥 Fake Video Detection")

uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    st.video(uploaded_video)

    # Placeholder for Fake Detection Logic
    st.subheader("🔍 Analyzing Video...")
    st.warning("⚠️ The video **might be fake**.")  # Example result

if st.button("🔙 Back to Home"):
    st.switch_page("app")
