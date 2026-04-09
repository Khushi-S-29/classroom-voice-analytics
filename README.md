# Classroom Voice Analytics

> **Audio → Transcript → Engagement Insights** for Indian classrooms  
> Offline-first · Indic language support · Acoustic diarization pipeline

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Approach](#approach)
3. [System Architecture](#system-architecture)
4. [How Metrics Are Calculated](#how-metrics-are-calculated)
5. [Tools and Models Used](#tools-and-models-used)
6. [How to Run](#how-to-run)
7. [Folder Structure](#folder-structure)
8. [Assumptions](#assumptions)
9. [Limitations](#limitations)

---

## Project Overview

Across India, what happens inside classrooms is largely invisible. Teachers receive little continuous feedback, student engagement goes unmeasured, and curriculum decisions are made without real classroom evidence.

This MVP converts classroom audio recordings into actionable engagement insights. It supports Hindi and other Indic languages, runs offline on a standard laptop, and produces a visual dashboard with labeled transcripts and four core engagement metrics.

---

## Approach

The pipeline has three stages that run in sequence (or in parallel for speed):

### Stage 1 — Transcription (Whisper)

Audio is passed to `faster-whisper`, an optimized implementation of OpenAI's Whisper model. The language is always set explicitly (e.g. `hi` for Hindi) rather than auto-detected, because auto-detection from noisy classroom intros frequently misidentifies Indic languages. An **initial prompt in the target language script** is also injected — this primes Whisper with Devanagari context and significantly reduces Latin-script hallucinations on Hindi audio.

Word-level timestamps are requested so each word can later be matched to a speaker.

### Stage 2 — Speaker Diarization (Pyannote)

`pyannote/speaker-diarization-3.1` runs on the same audio in parallel with Whisper. It identifies who is speaking at each moment in time, producing a list of speaker turns: `{SPEAKER_00: 0.0–6.2s, SPEAKER_01: 6.8–8.1s, ...}`.

No speaker count is forced — Pyannote's automatic clustering is used so it can correctly identify 2, 3, or more distinct voices. Forcing `num_speakers=2` causes all students to collapse into one cluster, which corrupts teacher identification.

### Stage 3 — Alignment and Role Assignment

Each word from Whisper is matched to a speaker turn from Pyannote using binary search (O(W log T)). Words that fall in gaps between turns are assigned to the nearest surrounding turn by time distance rather than being discarded.

Words are then merged into segments by consecutive speaker. The **teacher is identified** as the speaker with the highest combined score of `average_turn_duration × 0.6 + longest_turn × 0.4`. This is more reliable than total talk time because a group of students can collectively speak more than the teacher, but the teacher holds the floor longer per individual turn.

All segments shorter than 0.5 seconds are filtered out as noise before metrics are computed.

---

## System Architecture

```
Audio File (.wav / .mp3 / .m4a)
         │
         ├────────────────────────┐
         ▼                        ▼
┌─────────────────┐     ┌──────────────────────┐
│  faster-whisper │     │  pyannote 3.1        │
│  (Transcription)│     │  (Speaker Diarization)│
│                 │     │                       │
│  · lang forced  │     │  · num_speakers=auto  │
│  · initial_     │     │  · returns turns with │
│    prompt (hi)  │     │    speaker IDs        │
│  · word times   │     │                       │
└────────┬────────┘     └──────────┬────────────┘
         │                         │
         └──────────┬──────────────┘
                    ▼
         ┌─────────────────────┐
         │  Word-Speaker       │
         │  Alignment          │
         │  O(W log T) bisect  │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Teacher ID         │
         │  avg×0.6 + max×0.4  │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Engagement Metrics │
         │  TDR · SPI · ID · QRR│
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Streamlit Dashboard│
         │  Charts + Transcript│
         └─────────────────────┘
```

---

## How Metrics Are Calculated

All four metrics are computed in `src/analyzer.py` after speaker roles are assigned.

---

### 1. Teacher Dominance Ratio (TDR)

**What it measures:** How much of the spoken classroom time is occupied by the teacher.

```
TDR = Teacher Talk Time (seconds)
      ─────────────────────────────
      Total Spoken Time (seconds)
```

`Total Spoken Time = Teacher Talk Time + Student Talk Time`  
Silence is excluded from the denominator — TDR measures the balance of *speech*, not the proportion of a session spent talking.

| TDR Range | Interpretation |
|-----------|----------------|
| > 0.75 | Teacher-dominated / lecture-heavy |
| 0.55 – 0.70 | Ideal: teacher-led with student participation |
| < 0.50 | Student-led or highly interactive |

---

### 2. Student Participation Indicator (SPI)

**What it measures:** How actively students are participating, combining turn frequency with question-asking behaviour.

```
SPI = (Student Turns / Total Turns)  +  min(0.2,  0.05 × Student-Initiated Questions)
```

The base component counts what fraction of all speaker turns belong to students. The bonus rewards student-initiated questions (a stronger signal of active thinking than mere responses), capped at 0.2 to prevent saturation — a classroom with 20 student questions still produces a meaningful SPI rather than always showing 1.0.

| SPI Range | Interpretation |
|-----------|----------------|
| > 0.50 | Very high student participation |
| 0.30 – 0.50 | Ideal range |
| < 0.20 | Minimal student participation |

---

### 3. Interaction Density (ID)

**What it measures:** How frequently the conversation switches between teacher and student per minute of spoken time.

```
ID = Speaker Alternations
     ──────────────────────────
     Spoken Time (minutes)
```

A "speaker alternation" is any switch from TEACHER→STUDENT or STUDENT→TEACHER. The denominator is **spoken time in minutes**, not total session time — this makes ID comparable across sessions with different amounts of silence.

| ID Range | Interpretation |
|----------|----------------|
| > 8 / min | High-frequency back-and-forth dialogue |
| 4 – 8 / min | Moderate interaction |
| < 4 / min | Mostly monologue |

---

### 4. Question Response Rate (QRR)

**What it measures:** What fraction of teacher questions received a student response.

```
QRR = Student Response Segments
      ──────────────────────────────────  (capped at 1.0)
      Teacher Question Segments
```

A segment is classified as a question if it contains `?` or any of the question-word patterns in Hindi (`क्या, कौन, कब, कहाँ, क्यों, कैसे`) or English (`what, who, when, where, why, how`). If no teacher questions are detected, QRR is shown as **N/A** rather than 0.0, which would falsely imply students are unresponsive.

| QRR Range | Interpretation |
|-----------|----------------|
| > 0.80 | Strong student responsiveness |
| 0.50 – 0.80 | Moderate responsiveness |
| < 0.50 | Many questions going unanswered |

---

### Overall Engagement Level

A five-point rubric produces an overall label (High / Medium / Low):

```
+1 if TDR < 0.75   (teacher not fully dominating)
+1 if SPI > 0.25   (some student participation)
+1 if ID  > 5/min  (frequent exchanges)
+1 if TDR < 0.65   (good balance)
+1 if SPI > 0.35   (strong participation)

Score 4–5 → High Engagement
Score 2–3 → Medium Engagement
Score 0–1 → Low Engagement
```

---

## Tools and Models Used

| Component | Tool / Model | Version | Why |
|-----------|-------------|---------|-----|
| Transcription | faster-whisper | ≥ 1.0 | 4–8× faster than openai-whisper on CPU; same accuracy; supports 99 languages |
| Speaker Diarization | pyannote/speaker-diarization-3.1 | 3.1 | State-of-the-art open-source diarization; free with HuggingFace token |
| Dashboard | Streamlit | ≥ 1.32 | Rapid Python-native UI; no front-end code needed |
| Charts | Plotly | ≥ 5.18 | Interactive charts; works natively in Streamlit |
| Audio loading | torchaudio | — | Required by Pyannote for waveform input |
| Data export | pandas | ≥ 2.0 | CSV transcript export |
| Environment | python-dotenv | — | Load HF_TOKEN from .env file |

**Indic language support:** faster-whisper supports Hindi (`hi`), Marathi (`mr`), Tamil (`ta`), Bengali (`bn`), Telugu (`te`), Gujarati (`gu`), and 93 other languages out of the box.

---

## How to Run

### Prerequisites

```bash
pip install faster-whisper pyannote.audio streamlit plotly pandas \
            torchaudio python-dotenv
```

Also install `ffmpeg` for non-WAV audio formats:
```bash
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Ubuntu/Debian
```

### HuggingFace Token (required for Pyannote)

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the model license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Create a `.env` file in the project root:

```
HF_TOKEN=hf_your_token_here
```

### Run the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Folder Structure

```
classroom-voice-analytics/
├── app.py                  # Streamlit dashboard (main entry point)
├── requirements.txt
├── .env                    # HF_TOKEN (not committed to git)
├── .gitignore
├── README.md
└── src/
    ├── transcriber.py      # Whisper + Pyannote pipeline
    └── analyzer.py         # Metrics computation
```

---

## Assumptions

1. **Teacher is the dominant continuous speaker.** Assumes the teacher has the longest individual and average talk times. This may fail during extended student-led group discussions.  

2. **Single microphone recording.** Designed for one central mic capturing the entire room.

3. **Binary speaker model for roles.** Assigns only TEACHER or STUDENT. Co-teachers or visitors default to STUDENT unless they speak the most.

4. **Language set explicitly.** Auto language detection is unreliable when classroom audio begins with noise, music, or silence. The language hint (`hi` for Hindi) must be set in the sidebar.

5. **Minimum audio quality.** Pyannote requires reasonably clean audio to separate speakers. Very low bitrate recordings (< 32kbps), heavy background noise, or highly overlapping speech will degrade diarization quality.

6. **Questions detected by keywords.** Question detection uses Hindi and English question-word patterns. Rhetorical or implied questions without question words or `?` will not be detected.

---

## Limitations

1. **Pyannote on CPU is slow (~45–70s for 5-min audio).** Pyannote diarization is heavy, taking ~45–70s per 5 minutes of audio on a CPU. A GPU is highly recommended.

2. **No per-student identification.** All non-teacher speakers are labeled STUDENT. The system cannot distinguish between 30 different students or track individual student participation across time.

3. **Speaker overlap handling is basic.** When two people speak simultaneously, Pyannote assigns the dominant voice. The quieter voice's contribution is lost.

4. **Hindi transcription accuracy varies by accent and dialect.** faster-whisper `small` achieves approximately 85–90% word accuracy on standard Hindi. Regional dialects, code-switching (Hinglish), or heavy background noise will reduce accuracy. Use the `medium` or `large-v3` model for better results on challenging audio.

5. **Question detection is keyword-based.** The regex patterns cover standard question words but will miss indirect questions ("Tell me what you think about..."), implied questions, and non-standard phrasing.

6. **No longitudinal tracking.** Each audio file is analyzed independently. There is no mechanism to compare a teacher's engagement metrics across multiple sessions over time.

7. **No real-time streaming.** The pipeline processes complete audio files. Real-time classroom monitoring would require a streaming architecture that this MVP does not implement.



