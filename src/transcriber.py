import os, json, time
from typing import Optional


def transcribe_audio(
    audio_path: str,
    model_size: str = "tiny",
    language: Optional[str] = None,
) -> dict:
    try:
        return _transcribe_faster_whisper(audio_path, model_size, language)
    except ImportError:
        pass
    try:
        return _transcribe_openai_whisper(audio_path, model_size, language)
    except ImportError:
        raise ImportError(
            "No Whisper backend found.\n"
            "Fast (recommended): pip install faster-whisper\n"
            "Standard:           pip install openai-whisper"
        )



def _transcribe_faster_whisper(audio_path: str, model_size: str, language: Optional[str]) -> dict:
    from faster_whisper import WhisperModel  # pip install faster-whisper

    print(f"[faster-whisper] Loading '{model_size}' on CPU (int8)...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"[faster-whisper] Transcribing: {audio_path}")
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
    )
    if language:
        kwargs["language"] = language

    raw_segs, info = model.transcribe(audio_path, **kwargs)

    segments, full_text = [], []
    for seg in raw_segs:                    
        segments.append({
            "id":       seg.id,
            "start":    round(seg.start, 2),
            "end":      round(seg.end, 2),
            "text":     seg.text.strip(),
            "duration": round(seg.end - seg.start, 2),
        })
        full_text.append(seg.text.strip())

    print(f"[faster-whisper] Done in {time.time()-t0:.1f}s — lang: {info.language} (p={info.language_probability:.2f})")
    return {
        "text":       " ".join(full_text),
        "segments":   segments,
        "language":   info.language,
        "duration":   segments[-1]["end"] if segments else 0,
        "model":      f"faster-whisper/{model_size}",
        "audio_file": os.path.basename(audio_path),
    }



def _transcribe_openai_whisper(audio_path: str, model_size: str, language: Optional[str]) -> dict:
    import whisper  
    try:
        import torch
        on_gpu = torch.cuda.is_available()
    except ImportError:
        on_gpu = False

    print(f"[openai-whisper] Loading '{model_size}' on {'GPU' if on_gpu else 'CPU'}...")
    model = whisper.load_model(model_size)

    print(f"[openai-whisper] Transcribing: {audio_path}")
    t0 = time.time()

    opts = dict(
        fp16=on_gpu,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        verbose=False,
    )
    if language:
        opts["language"] = language

    result = model.transcribe(audio_path, **opts)
    print(f"[openai-whisper] Done in {time.time()-t0:.1f}s — lang: {result.get('language','?')}")

    segments = [
        {
            "id":       s["id"],
            "start":    round(s["start"], 2),
            "end":      round(s["end"], 2),
            "text":     s["text"].strip(),
            "duration": round(s["end"] - s["start"], 2),
        }
        for s in result.get("segments", [])
        if s.get("no_speech_prob", 0) <= 0.8
    ]

    return {
        "text":       result["text"].strip(),
        "segments":   segments,
        "language":   result.get("language", "unknown"),
        "duration":   segments[-1]["end"] if segments else 0,
        "model":      f"openai-whisper/{model_size}",
        "audio_file": os.path.basename(audio_path),
    }


def save_transcript(transcript: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    print(f"[Transcript] Saved → {path}")


def load_transcript_from_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



DEMO_TRANSCRIPT = {
    "audio_file": "demo_classroom_hindi.wav",
    "language":   "hi",
    "model":      "faster-whisper/tiny",
    "duration":   312.5,
    "text": (
        "नमस्ते बच्चों, आज हम भारत के स्वतंत्रता संग्राम के बारे में पढ़ेंगे। "
        "क्या आप सब तैयार हैं? हाँ मैडम। "
        "अच्छा, तो बताओ — 1857 का विद्रोह क्यों हुआ था? "
        "मैडम, अंग्रेजों के अत्याचार के कारण। "
        "और कौन था इस विद्रोह का नेतृत्व करने वाला? "
        "मंगल पांडे मैडम। झाँसी की रानी भी थीं। "
        "बहुत अच्छे! रानी लक्ष्मीबाई ने बहुत बहादुरी से लड़ाई की। "
        "अब मैं चाहती हूँ कि आप सब अपनी कॉपी में इन नामों को लिखें। "
        "क्या किसी को कुछ पूछना है? "
        "मैडम, क्या 1857 के बाद भारत आज़ाद हो गया? "
        "नहीं, 1857 एक शुरुआत थी। असली आज़ादी 1947 में मिली। "
        "समझे सब? हाँ मैडम। ठीक है, अब अगला पेज खोलो।"
    ),
    "segments": [
        {"id": 0,  "start": 0.0,  "end": 6.2,  "duration": 6.2,  "text": "नमस्ते बच्चों, आज हम भारत के स्वतंत्रता संग्राम के बारे में पढ़ेंगे।"},
        {"id": 1,  "start": 6.2,  "end": 9.8,  "duration": 3.6,  "text": "क्या आप सब तैयार हैं?"},
        {"id": 2,  "start": 10.5, "end": 12.0, "duration": 1.5,  "text": "हाँ मैडम।"},
        {"id": 3,  "start": 13.0, "end": 18.5, "duration": 5.5,  "text": "अच्छा, तो बताओ — 1857 का विद्रोह क्यों हुआ था?"},
        {"id": 4,  "start": 20.0, "end": 24.3, "duration": 4.3,  "text": "मैडम, अंग्रेजों के अत्याचार के कारण।"},
        {"id": 5,  "start": 25.0, "end": 30.0, "duration": 5.0,  "text": "और कौन था इस विद्रोह का नेतृत्व करने वाला?"},
        {"id": 6,  "start": 31.2, "end": 36.8, "duration": 5.6,  "text": "मंगल पांडे मैडम। झाँसी की रानी भी थीं।"},
        {"id": 7,  "start": 37.5, "end": 47.0, "duration": 9.5,  "text": "बहुत अच्छे! रानी लक्ष्मीबाई ने बहुत बहादुरी से लड़ाई की।"},
        {"id": 8,  "start": 48.0, "end": 58.0, "duration": 10.0, "text": "अब मैं चाहती हूँ कि आप सब अपनी कॉपी में इन नामों को लिखें।"},
        {"id": 9,  "start": 60.0, "end": 64.0, "duration": 4.0,  "text": "क्या किसी को कुछ पूछना है?"},
        {"id": 10, "start": 65.5, "end": 72.0, "duration": 6.5,  "text": "मैडम, क्या 1857 के बाद भारत आज़ाद हो गया?"},
        {"id": 11, "start": 73.0, "end": 82.0, "duration": 9.0,  "text": "नहीं, 1857 एक शुरुआत थी। असली आज़ादी 1947 में मिली।"},
        {"id": 12, "start": 83.0, "end": 86.0, "duration": 3.0,  "text": "समझे सब?"},
        {"id": 13, "start": 86.5, "end": 88.0, "duration": 1.5,  "text": "हाँ मैडम।"},
        {"id": 14, "start": 89.0, "end": 93.0, "duration": 4.0,  "text": "ठीक है, अब अगला पेज खोलो।"},
    ],
}