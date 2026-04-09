import os, time, logging, bisect
import torch
import concurrent.futures
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()
SYSTEM_HF_TOKEN = os.environ.get("HF_TOKEN", "")


if SYSTEM_HF_TOKEN:
    from huggingface_hub import login
    login(token=SYSTEM_HF_TOKEN)
    print("LOGGED IN VIA ENV TOKEN")
else:
    print("NO ENV TOKEN FOUND")

def get_diarization_pipeline(hf_token: str, device_type: str):
    """Load pyannote pipeline once and cache it."""
    try:
        import streamlit as st
        @st.cache_resource(show_spinner=False)
        def _load(token, device):
            from pyannote.audio import Pipeline
            logger.info(f"[Pyannote] Loading model on {device.upper()} (first time only)...")
            t0 = time.time()
            pipe = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token
            ).to(torch.device(device))
            if hasattr(pipe, '_segmentation'):
                pipe._segmentation.batch_size = 8
            logger.info(f"[Pyannote] Model loaded in {time.time()-t0:.1f}s")
            return pipe
        return _load(hf_token, device_type)
    except Exception:
        from pyannote.audio import Pipeline
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        ).to(torch.device(device_type))
        return pipe


def get_whisper_model(model_size: str, device_type: str, compute_type: str):
    try:
        import streamlit as st
        @st.cache_resource(show_spinner=False)
        def _load(size, device, ctype):
            from faster_whisper import WhisperModel
            logger.info(f"[Whisper] Loading {size} on {device.upper()} (first time only)...")
            t0 = time.time()
            m = WhisperModel(size, device=device, compute_type=ctype)
            logger.info(f"[Whisper] Model loaded in {time.time()-t0:.1f}s")
            return m
        return _load(model_size, device_type, compute_type)
    except Exception:
        from faster_whisper import WhisperModel
        return WhisperModel(model_size, device=device_type, compute_type=compute_type)


def run_diarization(audio_path, hf_token, device_type, num_speakers=None):
    pipeline = get_diarization_pipeline(hf_token, device_type)
    logger.info("[Pyannote] Running diarization...")
    t0 = time.time()

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers

    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)

    diarization = pipeline(
    {
        "waveform": waveform,
        "sample_rate": sample_rate
    },
    **kwargs)
    turns = [
        {"start": turn.start, "end": turn.end, "speaker_id": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    logger.info(f"[Pyannote] Done in {time.time()-t0:.1f}s — {len(turns)} turns")
    return turns


def run_transcription(audio_path, model_size, device_type, compute_type, language):
    model = get_whisper_model(model_size, device_type, compute_type)
    logger.info(f"[Whisper] Transcribing with {model_size}...")
    t0 = time.time()

    kwargs = dict(
        beam_size=1,                      
        best_of=1,
        temperature=0.0,                    
        condition_on_previous_text=False,   
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        vad_filter=True,                    
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True,              
    )
    if language:
        kwargs["language"] = language

    segments_gen, info = model.transcribe(audio_path, **kwargs)

    words, last_end = [], 0
    for seg in segments_gen:
        for w in seg.words:
            words.append({"start": w.start, "end": w.end, "text": w.word})
            last_end = w.end

    logger.info(f"[Whisper] Done in {time.time()-t0:.1f}s — {len(words)} words, lang={info.language}")
    return words, info, last_end



def align_words_to_speakers(words, diarization_turns):
   
    if not diarization_turns:
        for w in words:
            w["speaker"] = "UNKNOWN"
        return

    starts = [t["start"] for t in diarization_turns]

    for word in words:
        midpoint = (word["start"] + word["end"]) / 2
        idx = bisect.bisect_right(starts, midpoint) - 1
        if idx >= 0 and diarization_turns[idx]["end"] >= midpoint:
            word["speaker"] = diarization_turns[idx]["speaker_id"]
        else:
            word["speaker"] = "UNKNOWN"


def words_to_segments(words):
    segments, current = [], None
    for word in words:
        if current is None or current["speaker_id"] != word["speaker"]:
            if current:
                current["text"]     = " ".join(current["_words"]).strip()
                current["duration"] = round(current["end"] - current["start"], 2)
                del current["_words"]
                segments.append(current)
            current = {
                "start":      round(word["start"], 2),
                "end":        round(word["end"], 2),
                "speaker_id": word["speaker"],
                "_words":     [word["text"].strip()],
            }
        else:
            current["end"] = round(word["end"], 2)
            current["_words"].append(word["text"].strip())

    if current:
        current["text"]     = " ".join(current["_words"]).strip()
        current["duration"] = round(current["end"] - current["start"], 2)
        del current["_words"]
        segments.append(current)

    return segments


def transcribe_audio(
    audio_path: str,
    hf_token: str,
    model_size: str = "small",             
    language: Optional[str] = None,
    parallel: bool = True,
    num_speakers: Optional[int] = None,    
) -> dict:

    if not hf_token:
        raise ValueError("A Hugging Face token is required for Pyannote diarization.")

    device_type  = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device_type == "cuda" else "int8"

    logger.info(f"[Pipeline] Device={device_type.upper()} | Model={model_size} | Parallel={parallel}")
    t_total = time.time()

    if parallel:
        logger.info("[Pipeline] Running diarization + transcription concurrently...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_diar  = ex.submit(run_diarization, audio_path, hf_token, device_type, num_speakers)
            f_trans = ex.submit(run_transcription, audio_path, model_size, device_type, compute_type, language)
            diarization_turns   = f_diar.result()
            words, info, last_end = f_trans.result()
    else:
        diarization_turns         = run_diarization(audio_path, hf_token, device_type, num_speakers)
        words, info, last_end     = run_transcription(audio_path, model_size, device_type, compute_type, language)

    logger.info("[Alignment] Merging word timestamps with speaker turns...")
    align_words_to_speakers(words, diarization_turns)

    merged_segments = words_to_segments(words)

    speaker_durations = {}
    for seg in merged_segments:
        spk = seg["speaker_id"]
        if spk != "UNKNOWN":
            speaker_durations[spk] = speaker_durations.get(spk, 0) + seg["duration"]

    teacher_id = max(speaker_durations, key=speaker_durations.get) if speaker_durations else "UNKNOWN"
    logger.info(f"[Pipeline] Teacher auto-assigned: {teacher_id} "
                f"({speaker_durations.get(teacher_id, 0):.0f}s of speech)")

    for seg in merged_segments:
        seg["role"] = "TEACHER" if seg["speaker_id"] == teacher_id else "STUDENT"

    logger.info(f"[Pipeline] Total time: {time.time()-t_total:.1f}s")

    return {
        "text":       " ".join(w["text"] for w in words),
        "segments":   merged_segments,
        "language":   info.language,
        "duration":   last_end,
        "model":      f"faster-whisper/{model_size} + pyannote-3.1",
        "audio_file": os.path.basename(audio_path),
        "teacher_id": teacher_id,
    }