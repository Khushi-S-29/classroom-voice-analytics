import re
from typing import List, Dict


_RE_STUDENT_ADDR = re.compile(
    r'\b(मैडम|दीदी|madam|ma\'am|sir|सर)\b', re.IGNORECASE
)

_STUDENT_EXACT = frozenset([
    "हाँ", "हां", "जी", "हाँ जी", "जी हाँ",
    "yes", "no", "okay", "ok", "hmm", "ji",
])

_RE_TEACHER = re.compile(
    r'(बताओ|लिखो|खोलो|सुनो|पढ़ो|देखो|समझे|याद करो|चाहती|चाहता|'
    r'शाबाश|बिल्कुल सही|बहुत अच्छे|ठीक है|नमस्ते|आज हम|'
    r'open your|turn to page|very good|excellent|well done|'
    r'write down|pay attention|today we|let\'s begin|repeat after)',
    re.IGNORECASE
)

_RE_TEACHER_NEGATION = re.compile(r'^(नहीं|no)[,،]\s*\S+\s+\S+\s+\S+\s+\S+', re.IGNORECASE)

_RE_QUESTION = re.compile(
    r'(क्या|कौन|कब|कहाँ|क्यों|कैसे|कितना|किसने|'
    r'\bwhat\b|\bwho\b|\bwhen\b|\bwhere\b|\bwhy\b|\bhow\b|\bwhich\b)',
    re.IGNORECASE
)



def classify_speaker(text: str, prev_speaker: str = "TEACHER") -> str:
    t  = text.strip()
    tl = t.lower()
    wc = len(t.split())

    if _RE_STUDENT_ADDR.search(t):
        return "STUDENT"

    if _RE_TEACHER_NEGATION.match(t):
        return "TEACHER"

    if _RE_TEACHER.search(t):
        return "TEACHER"

    if tl in _STUDENT_EXACT:
        return "STUDENT"

    if wc <= 3 and prev_speaker == "TEACHER":
        return "STUDENT"

    if wc > 15:
        return "TEACHER"

    return prev_speaker



def is_question(text: str) -> bool:
    return "?" in text or bool(_RE_QUESTION.search(text))


def detect_silences(segments: List[Dict], min_gap: float = 1.5) -> List[Dict]:
    return [
        {
            "start":    round(segments[i - 1]["end"], 2),
            "end":      round(segments[i]["start"], 2),
            "duration": round(segments[i]["start"] - segments[i - 1]["end"], 2),
        }
        for i in range(1, len(segments))
        if segments[i]["start"] - segments[i - 1]["end"] >= min_gap
    ]

def analyze_transcript(transcript: dict) -> dict:
    segments       = transcript.get("segments", [])
    total_duration = transcript.get("duration", 0) or 1

    labeled, prev_speaker = [], "TEACHER"
    for seg in segments:
        speaker  = classify_speaker(seg["text"], prev_speaker)
        question = is_question(seg["text"])
        labeled.append({**seg, "speaker": speaker, "is_question": question})
        prev_speaker = speaker

    teacher_time = sum(s["duration"] for s in labeled if s["speaker"] == "TEACHER")
    student_time = sum(s["duration"] for s in labeled if s["speaker"] == "STUDENT")
    spoken_time  = teacher_time + student_time or 1

    silences     = detect_silences(segments)
    silence_time = sum(s["duration"] for s in silences)

    teacher_questions = [s for s in labeled if s["speaker"] == "TEACHER" and s["is_question"]]
    student_questions = [s for s in labeled if s["speaker"] == "STUDENT" and s["is_question"]]
    student_responses = [s for s in labeled if s["speaker"] == "STUDENT" and not s["is_question"]]

    alternations = sum(
        1 for i in range(1, len(labeled))
        if labeled[i]["speaker"] != labeled[i - 1]["speaker"]
    )

    tdr = round(teacher_time / spoken_time, 3)

    total_turns   = len(labeled) or 1
    student_turns = sum(1 for s in labeled if s["speaker"] == "STUDENT")
    spi = round(min(1.0, student_turns / total_turns + 0.05 * len(student_questions)), 3)

    duration_min        = total_duration / 60 or 1
    interaction_density = round(alternations / duration_min, 2)

    qrr = round(
        min(1.0, len(student_responses) / len(teacher_questions))
        if teacher_questions else 0.0,
        3
    )

    score = sum([
        tdr < 0.75,
        spi > 0.25,
        interaction_density > 5,
        tdr < 0.65,
        spi > 0.35,
    ])
    engagement = "High" if score >= 4 else "Medium" if score >= 2 else "Low"

    t_pct = int(100 * teacher_time / total_duration)
    s_pct = int(100 * student_time / total_duration)
    parts = [
        f"This session shows {engagement.lower()} student engagement.",
        f"Teacher spoke {t_pct}% of the session; students contributed {s_pct}%.",
        f"Teacher posed {len(teacher_questions)} question(s) with {len(student_responses)} student response(s).",
    ]
    if student_questions:
        parts.append(f"Students asked {len(student_questions)} question(s) — strong curiosity signal.")
    else:
        parts.append("No student-initiated questions detected — encouraging Qs could deepen engagement.")
    if tdr > 0.75:
        parts.append("TDR is high; consider more student-led discussion or pair work.")
    if silence_time > 10:
        parts.append(f"{int(silence_time)}s of silence detected — may reflect pauses for reflection.")

    return {
        "labeled_segments": labeled,
        "silences":         silences,
        "stats": {
            "total_duration_s":    round(total_duration, 1),
            "teacher_talk_time_s": round(teacher_time, 1),
            "student_talk_time_s": round(student_time, 1),
            "silence_time_s":      round(silence_time, 1),
            "teacher_talk_pct":    round(100 * teacher_time / total_duration, 1),
            "student_talk_pct":    round(100 * student_time / total_duration, 1),
            "silence_pct":         round(100 * silence_time / total_duration, 1),
            "total_segments":      len(labeled),
            "teacher_questions":   len(teacher_questions),
            "student_questions":   len(student_questions),
            "student_responses":   len(student_responses),
            "interaction_count":   alternations,
        },
        "metrics": {
            "teacher_dominance_ratio":         tdr,
            "student_participation_indicator": spi,
            "interaction_density":             interaction_density,
            "question_response_rate":          qrr,
        },
        "engagement_level": engagement,
        "summary":          " ".join(parts),
        "language":         transcript.get("language", "unknown"),
    }