import os
import tempfile
import streamlit as st
from docsum_audio import run_pipeline

st.set_page_config(page_title="Extractive Audio Summarizer", layout="centered")
st.title("Extractive Audio Summarizer")
st.write("Upload an interview or talk. We'll cut an **extractive** summary and output a smoothed audio edit.")

uploaded = st.file_uploader("Upload audio (WAV/MP3/M4A/FLAC)", type=["wav", "mp3", "m4a", "flac"]) 

col1, col2 = st.columns(2)
with col1:
    mode = st.selectbox("Target length mode", ["Auto", "Ratio (e.g., 0.07)", "Fixed minutes"]) 
with col2:
    language = st.selectbox("Language", ["auto", "en"])  # MVP: en or auto

crossfade_ms = st.slider("Crossfade (ms)", 0, 300, 90, 10)
gap_ms = st.slider("Gap between sentences (ms)", 0, 500, 150, 10)

ratio_val = 0.07
minutes_val = 4
if mode == "Ratio (e.g., 0.07)":
    ratio_val = st.number_input("Target ratio of original duration", min_value=0.01, max_value=0.9, value=0.07, step=0.01)
elif mode == "Fixed minutes":
    minutes_val = st.number_input("Target minutes", min_value=1, max_value=60, value=4, step=1)

model_size = st.selectbox("Transcription model", ["small", "medium"], index=0)

if uploaded and st.button("Summarize"):
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    target_ratio = 0.07
    if mode == "Ratio (e.g., 0.07)":
        target_ratio = float(ratio_val)
    elif mode == "Fixed minutes":
        from pydub import AudioSegment
        dur_s = len(AudioSegment.from_file(tmp_path)) / 1000.0
        target_ratio = max(0.01, min(0.9, (minutes_val * 60.0) / max(1.0, dur_s)))

    lang_arg = None if language == "auto" else language

    with st.spinner("Processing… this can take a while for long files."):
        run_pipeline(
            input_path=tmp_path,
            target_ratio=target_ratio,
            model_size=model_size,
            language=lang_arg,
            crossfade_ms=int(crossfade_ms),
            gap_ms=int(gap_ms),
            use_vad=True,
        )

    base, _ = os.path.splitext(tmp_path)
    out_wav = f"{base}_summary.wav"
    out_txt = f"{base}_summary.txt"
    out_json = f"{base}_summary.cues.json"

    if os.path.exists(out_wav):
        st.success("Done! Download your files below.")
        with open(out_wav, "rb") as f:
            st.audio(f.read(), format="audio/wav")
        with open(out_wav, "rb") as f:
            st.download_button("Download summarized audio (.wav)", f, file_name=os.path.basename(out_wav))
    if os.path.exists(out_txt):
        with open(out_txt, "r", encoding="utf-8") as f:
            st.download_button("Download kept transcript (.txt)", f, file_name=os.path.basename(out_txt))
    if os.path.exists(out_json):
        with open(out_json, "rb") as f:
            st.download_button("Download cue JSON (.json)"
