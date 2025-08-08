import streamlit as st
from docsum_audio import process_audio

st.title("Audio Summarizer (VAD-Free)")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "flac"])

if uploaded_file:
    with open("temp_input.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Processing...")
    output_path, summary_segments = process_audio("temp_input.wav")

    st.audio(output_path)
    st.write("Summary Segments:")
    for start, end, text in summary_segments:
        st.write(f"[{start:.2f}s - {end:.2f}s]: {text}")
