import os
import tempfile
from faster_whisper import WhisperModel
from pydub import AudioSegment
import soundfile as sf

def process_audio(file_path):
    """
    Takes in an audio file path, transcribes it with faster-whisper,
    and returns a summary audio file path + list of (start, end, text) segments.
    """

    # Convert to wav if not already
    base, ext = os.path.splitext(file_path)
    if ext.lower() != ".wav":
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        AudioSegment.from_file(file_path).export(temp_wav.name, format="wav")
        file_path = temp_wav.name

    # Load faster-whisper model (small = faster)
    model = WhisperModel("small", device="cpu", compute_type="int8")

    # Transcribe
    segments, _ = model.transcribe(file_path, beam_size=5)

    # Collect transcript segments
    summary_segments = []
    combined_text = []
    for segment in segments:
        text = segment.text.strip()
        summary_segments.append((segment.start, segment.end, text))
        combined_text.append(text)

    # Create summary WAV (spoken summary text)
    summary_text = " ".join(combined_text)
    tts_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(tts_audio_path, AudioSegment.silent(duration=500).get_array_of_samples(), 16000)  # Placeholder audio

    return tts_audio_path, summary_segments
