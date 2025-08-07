#!/usr/bin/env python3
import argparse
import json
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from pydub import AudioSegment, effects
from pydub.utils import mediainfo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Transcription
from faster_whisper import WhisperModel

# Optional: voice activity detection to nudge cut points off breaths/noise
import webrtcvad
import soundfile as sf

@dataclass
class Word:
    text: str
    start: float
    end: float

@dataclass
class Sentence:
    text: str
    start: float
    end: float
    words: List[Word]

SENTENCE_PUNCT = {".", "?", "!"}

def transcribe(input_path: str, model_size: str = "small", language: Optional[str] = None) -> List[Word]:
    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, info = model.transcribe(input_path, language=language, vad_filter=True, word_timestamps=True)
    words: List[Word] = []
    for seg in segments:
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            words.append(Word(text=w.word.strip(), start=float(w.start), end=float(w.end)))
    return words

def words_to_sentences(words: List[Word], max_gap_s: float = 1.2, min_chars: int = 12) -> List[Sentence]:
    sents: List[Sentence] = []
    cur_words: List[Word] = []
    for i, w in enumerate(words):
        if not w.text:
            continue
        cur_words.append(w)
        is_boundary = False
        if w.text.endswith(tuple(SENTENCE_PUNCT)):
            is_boundary = True
        else:
            if i < len(words) - 1 and (words[i+1].start - w.end) > max_gap_s:
                is_boundary = True
        if is_boundary and cur_words:
            txt = " ".join([cw.text for cw in cur_words]).strip()
            if len(txt) >= min_chars:
                sents.append(Sentence(text=txt, start=cur_words[0].start, end=cur_words[-1].end, words=cur_words.copy()))
            cur_words = []
    if cur_words:
        txt = " ".join([cw.text for cw in cur_words]).strip()
        if len(txt) >= min_chars:
            sents.append(Sentence(text=txt, start=cur_words[0].start, end=cur_words[-1].end, words=cur_words.copy()))
    return sents

def rank_sentences(sent_texts: List[str]) -> np.ndarray:
    if not sent_texts:
        return np.array([])
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sent_texts)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)
    G = nx.from_numpy_array(sim)
    pr = nx.pagerank(G, alpha=0.85)
    scores = np.array([pr[i] for i in range(len(sent_texts))])
    return scores

def mmr_selection(sent_texts: List[str], base_scores: np.ndarray, target_k: int, diversity: float = 0.6) -> List[int]:
    if target_k >= len(sent_texts):
        return list(range(len(sent_texts)))
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sent_texts)
    selected: List[int] = []
    candidate = set(range(len(sent_texts)))
    first = int(np.argmax(base_scores))
    selected.append(first)
    candidate.remove(first)
    while len(selected) < target_k and candidate:
        sim_to_selected = cosine_similarity(X[list(candidate)], X[selected]).max(axis=1)
        cand_list = list(candidate)
        utilities = diversity * base_scores[cand_list] - (1 - diversity) * sim_to_selected
        pick = cand_list[int(np.argmax(utilities))]
        selected.append(pick)
        candidate.remove(pick)
    return sorted(selected)

from scipy.signal import resample_poly
from math import gcd

def load_pcm_mono(path: str, target_rate: int = 16000):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_rate:
        g = gcd(sr, target_rate)
        up = target_rate // g
        down = sr // g
        data = resample_poly(data, up, down)
        sr = target_rate
    data = np.clip(data, -1.0, 1.0)
    pcm16 = (data * 32767).astype(np.int16)
    return pcm16, sr

def find_nearest_silence(pcm16: np.ndarray, sr: int, t_sec: float, window_ms: int = 30, search_ms: int = 300, mode: int = 2) -> float:
    vad = webrtcvad.Vad(mode)
    frame_len = int(sr * (window_ms / 1000.0))
    step = frame_len
    center = int(t_sec * sr)
    radius = int(sr * (search_ms / 1000.0))
    best_idx = center
    best_score = 1e9
    for idx in range(max(0, center - radius), min(len(pcm16) - frame_len, center + radius), step):
        frame = pcm16[idx: idx + frame_len].tobytes()
        voiced = vad.is_speech(frame, sr)
        energy = float(np.mean((pcm16[idx: idx + frame_len].astype(np.float32)) ** 2))
        score = (1000 if voiced else 0) + energy
        if score < best_score:
            best_score = score
            best_idx = idx
    return best_idx / sr

def assemble_audio(input_path: str, spans: List[Tuple[float, float]], crossfade_ms: int = 60, gap_ms: int = 120, normalize_lufs: float = -16.0) -> AudioSegment:
    audio = AudioSegment.from_file(input_path)
    pieces = []
    for (s, e) in spans:
        s_ms = max(0, int(s * 1000))
        e_ms = max(0, int(e * 1000))
        if e_ms > s_ms:
            seg = audio[s_ms:e_ms]
            seg = seg.fade_in(10).fade_out(10)
            pieces.append(seg)
    if not pieces:
        return AudioSegment.silent(duration=500)
    out = pieces[0]
    for seg in pieces[1:]:
        if gap_ms > 0:
            out = out.append(AudioSegment.silent(gap_ms), crossfade=crossfade_ms)
        out = out.append(seg, crossfade=crossfade_ms)
    if normalize_lufs != 0:
        out = effects.normalize(out)
    return out

def run_pipeline(
    input_path: str,
    target_ratio: float = 0.07,
    max_sentences: Optional[int] = None,
    model_size: str = "small",
    language: Optional[str] = None,
    crossfade_ms: int = 60,
    gap_ms: int = 120,
    use_vad: bool = True,
    normalize_lufs: float = -16.0,
):
    words = transcribe(input_path, model_size=model_size, language=language)
    if not words:
        raise RuntimeError("No words found in transcription.")
    sentences = words_to_sentences(words)
    sent_texts = [s.text for s in sentences]
    base_scores = rank_sentences(sent_texts)
    avg_dur = np.mean([s.end - s.start for s in sentences]) if sentences else 3.0
    audio_info = mediainfo(input_path)
    total_dur = float(audio_info.get("duration", 0))
    desired_duration = total_dur * target_ratio
    k_by_ratio = max(1, int(math.ceil(desired_duration / max(1e-6, avg_dur))))
    target_k = max_sentences if max_sentences else k_by_ratio
    picked_idx = mmr_selection(sent_texts, base_scores, target_k=target_k, diversity=0.6)
    picked = [sentences[i] for i in picked_idx]
    spans = []
    pcm16 = None
    sr = 16000
    if use_vad:
        try:
            pcm16, sr = load_pcm_mono(input_path, target_rate=16000)
        except Exception:
            use_vad = False
    for s in picked:
        start, end = s.start, s.end
        if use_vad and pcm16 is not None:
            start = find_nearest_silence(pcm16, sr, start, window_ms=30, search_ms=300, mode=2)
            end = find_nearest_silence(pcm16, sr, end, window_ms=30, search_ms=300, mode=2)
            if end <= start:
                end = s.end
        spans.append((max(0.0, start), max(start + 0.05, end)))
    out_audio = assemble_audio(input_path, spans, crossfade_ms=crossfade_ms, gap_ms=gap_ms, normalize_lufs=normalize_lufs)
    base, ext = os.path.splitext(input_path)
    out_wav = f"{base}_summary.wav"
    out_txt = f"{base}_summary.txt"
    out_json = f"{base}_summary.cues.json"
    out_audio.export(out_wav, format="wav")
    kept_text = "\n\n".join([s.text for s in picked])
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(kept_text)
    cues = [
        {"text": s.text, "start": float(spans[i][0]), "end": float(spans[i][1])}
        for i, s in enumerate(picked)
    ]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"spans": cues}, f, ensure_ascii=False, indent=2)
    print("Done.")
    print(out_wav)
    print(out_txt)
    print(out_json)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extractive audio summarizer → audio cut & smoothed output.")
    ap.add_argument("--input", required=True, help="Path to input audio (wav/mp3/m4a/flac)")
    ap.add_argument("--target-ratio", type=float, default=0.07, help="Fraction of duration to keep, e.g., 0.07 ≈ 4 min from 60 min")
    ap.add_argument("--max-sentences", type=int, default=None, help="Optional hard cap on number of kept sentences")
    ap.add_argument("--model", default="medium", help="Whisper model size: tiny|base|small|medium|large-v2")
    ap.add_argument("--language", default=None, help="Force language code (e.g., en, es). If omitted, auto-detect")
    ap.add_argument("--crossfade-ms", type=int, default=60, help="Crossfade length between kept chunks")
    ap.add_argument("--gap-ms", type=int, default=120, help="Optional short silence between sentences")
    ap.add_argument("--no-vad", action="store_true", help="Disable VAD boundary nudging")
    ap.add_argument("--normalize-lufs", type=float, default=-16.0, help="Set 0 to skip normalization")
    args = ap.parse_args()
    run_pipeline(
        input_path=args.input,
        target_ratio=args.target_ratio,
        max_sentences=args.max_sentences,
        model_size=args.model,
        language=args.language,
        crossfade_ms=args.crossfade_ms,
        gap_ms=args.gap_ms,
        use_vad=(not args.no_vad),
        normalize_lufs=args.normalize_lufs,
    )
