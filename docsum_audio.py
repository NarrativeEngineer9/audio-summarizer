import os
import tempfile
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from pydub import AudioSegment

def transcribe_audio(audio_path):
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    return [(seg.start, seg.end, seg.text) for seg in segments]

def summarize_segments(segments, ratio=0.3):
    total_segments = len(segments)
    keep_count = max(1, int(total_segments * ratio))
    return segments[:keep_count]

def export_summary_audio(original_path, summary_segments):
    audio = AudioSegment.from_file(original_path)
    combined = AudioSegment.silent(duration=0)

    for start, end, _ in summary_segments:
        seg_audio = audio[start * 1000:end * 1000]
        combined += seg_audio

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        combined.export(tmp.name, format="wav")
        return tmp.name

def process_audio(file_path):
    segments = transcribe_audio(file_path)
    summary_segments = summarize_segments(segments)
    output_path = export_summary_audio(file_path, summary_segments)
    return output_path, summary_segments
