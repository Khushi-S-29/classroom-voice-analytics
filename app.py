import os, sys, tempfile, logging
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

load_dotenv()

SYSTEM_HF_TOKEN = os.environ.get("HF_TOKEN", "")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from analyzer import analyze_transcript

st.set_page_config(page_title="Classroom Voice Analytics", page_icon="🏫", layout="wide")

if SYSTEM_HF_TOKEN:
    from huggingface_hub import login
    login(token=SYSTEM_HF_TOKEN)
    print("LOGGED IN VIA ENV TOKEN")
else:
    print("NO ENV TOKEN FOUND")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.main { background: #0d1117; }
.block-container { padding: 1.5rem 2rem; }
.metric-card {
    background: #161d2d; border-radius: 12px; padding: 1.2rem 1.4rem;
    border: 1px solid rgba(99,179,237,0.15); margin-bottom: 0.75rem;
}
.metric-label { color: #718096; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { color: #e2e8f0; font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.metric-sub   { color: #4a5568; font-size: 0.76rem; margin-top: 0.2rem; }
.summary-box {
    background: linear-gradient(135deg, #0d2136 0%, #0d1117 100%);
    border-left: 4px solid #63b3ed; border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem; margin: 1rem 0;
    color: #a0aec0; font-size: 0.92rem; line-height: 1.7;
}
.badge-high   { background:#1a4731; color:#68d391; border:1px solid #48bb78; }
.badge-medium { background:#7b4f1a; color:#f6ad55; border:1px solid #ed8936; }
.badge-low    { background:#4a1a1a; color:#fc8181; border:1px solid #f56565; }
.engagement-badge {
    display:inline-block; padding:0.25rem 0.9rem; border-radius:999px;
    font-size:0.8rem; font-weight:600; letter-spacing:0.05em;
}
.seg-teacher { background:#0a1e35; border-left:3px solid #63b3ed; padding:0.55rem 0.9rem; margin:0.25rem 0; border-radius:0 8px 8px 0; font-size:0.87rem; }
.seg-student { background:#0d2b1a; border-left:3px solid #68d391; padding:0.55rem 0.9rem; margin:0.25rem 0; border-radius:0 8px 8px 0; font-size:0.87rem; }
.seg-meta { color:#4a5568; font-size:0.68rem; font-family:'JetBrains Mono',monospace; margin-right:0.4rem; }
.section-hdr {
    color:#4a5568; font-size:0.68rem; text-transform:uppercase;
    letter-spacing:0.1em; font-weight:600;
    margin:1.5rem 0 0.75rem; padding-bottom:0.4rem; border-bottom:1px solid #1a202c;
}
section[data-testid="stSidebar"] { background: #0a0f1a; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#1a2744 0%,#0f1923 100%);
     border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem;
     border:1px solid rgba(99,179,237,0.2);">
  <div style="display:flex;align-items:center;gap:1rem;">
    <span style="font-size:2.2rem;"></span>
    <div>
      <h1 style="color:#e2e8f0;font-size:1.9rem;font-weight:700;margin:0 0 0.2rem;"> Classroom Voice Analytics</h1>
      <p style="color:#718096;font-size:0.9rem;margin:0;">Acoustic Pipeline · Pyannote Diarization + Faster-Whisper · Indic Language Support</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Pipeline Settings")

    if SYSTEM_HF_TOKEN:
        hf_token = SYSTEM_HF_TOKEN
    else:
        hf_token = st.text_input("Hugging Face Token", type="password",
                                  help="Required for Pyannote diarization. Get one at huggingface.co")
        if not hf_token:
            st.caption("Get a free token at [huggingface.co](https://huggingface.co/settings/tokens)")

    st.divider()

    whisper_model = st.selectbox(
        "Whisper Model",
        ["small", "medium", "large-v3"],
        index=0,
        help="small=fastest+great Hindi · medium=balanced · large-v3=most accurate but slow on CPU",
    )

    speed_map = {"small": "~10-25s", "medium": "~25-50s", "large-v3": "~50-120s"}
    st.caption(f"Whisper est. (5-min audio, CPU): **{speed_map[whisper_model]}**")
    st.caption("Pyannote always adds ~40-70s on CPU regardless of Whisper model.")

    lang_choice = st.selectbox(
        "Language Hint",
        ["Auto-detect", "Hindi (hi)", "Marathi (mr)", "Tamil (ta)",
         "Bengali (bn)", "Telugu (te)", "Gujarati (gu)", "English (en)"],
    )
    lang_code = None if lang_choice == "Auto-detect" else lang_choice.split("(")[1].rstrip(")")

    st.divider()
    st.markdown("###  Performance")

    run_parallel = st.checkbox(
        "Parallel Mode",
        value=True,
        help="Runs Pyannote + Whisper simultaneously. Saves 10-25s but uses more RAM.",
    )

    speaker_count = st.selectbox(
        "Number of speakers",
        ["Auto-detect", "2 (typical classroom)", "3", "4", "5+"],
        index=1,
        help="Setting this skips expensive clustering and saves 5-10s in Pyannote.",
    )
    num_speakers = None if speaker_count == "Auto-detect" or speaker_count == "5+" else int(speaker_count.split()[0])

    st.divider()
    st.markdown("""
    <div style="color:#4a5568;font-size:0.72rem;line-height:1.7;">
    <strong style="color:#718096">Speed tips</strong><br>
    • Models are <strong>cached</strong> after first load<br>
    • Set speaker count to skip clustering<br>
    • Use GPU for 10x speedup on Pyannote<br>
    • <code>small</code> model is 4x faster than <code>large-v3</code>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload Audio", type=["wav", "mp3", "m4a", "ogg", "flac"])

transcript = analysis = error_msg = None

if uploaded_file and hf_token:
    logger.info(f"--- NEW JOB: {uploaded_file.name} | model={whisper_model} | parallel={run_parallel} ---")

    with st.status(" Running acoustic pipeline...", expanded=True) as status:
        try:
            from transcriber import transcribe_audio

            st.write("📁 Saving audio to temp file...")
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            mode_str = "parallel" if run_parallel else "sequential"
            spk_str  = str(num_speakers) if num_speakers else "auto"
            st.write(f" {mode_str.capitalize()} pipeline · {whisper_model} · {spk_str} speakers")

            transcript = transcribe_audio(
                audio_path=tmp_path,
                hf_token=hf_token,
                model_size=whisper_model,
                language=lang_code,
                parallel=run_parallel,
                num_speakers=num_speakers,
            )
            os.unlink(tmp_path)

            st.write("Calculating engagement metrics...")
            analysis = analyze_transcript(transcript)

            lang_detected = transcript.get("language", "?").upper()
            teacher_id    = transcript.get("teacher_id", "?")
            status.update(
                label=f"✅ Done — lang={lang_detected} · teacher={teacher_id}",
                state="complete", expanded=False
            )
            logger.info("--- JOB COMPLETE ---")

        except Exception as e:
            status.update(label="❌ Pipeline failed", state="error", expanded=True)
            logger.error(f"Pipeline error: {e}", exc_info=True)
            error_msg = str(e)

elif uploaded_file and not hf_token:
    st.warning(" Add your Hugging Face token in the sidebar or a `.env` file (`HF_TOKEN=...`) to begin.")

if error_msg:
    st.error(f"**Error:** {error_msg}")
    if "token" in error_msg.lower() or "401" in error_msg:
        st.info("Make sure your HF token has accepted the [Pyannote model license](https://huggingface.co/pyannote/speaker-diarization-3.1).")

if analysis:
    stats   = analysis["stats"]
    metrics = analysis["metrics"]
    segs    = analysis["labeled_segments"]
    level   = analysis["engagement_level"]

    st.markdown('<div class="section-hdr">Session Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    def metric_card(col, label, value, sub=""):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    dur_min = int(stats["total_duration_s"] // 60)
    dur_sec = int(stats["total_duration_s"] % 60)
    metric_card(c1, "Duration",         f"{dur_min}m {dur_sec}s",          f"{stats['total_segments']} segments")
    metric_card(c2, "Teacher Talk",     f"{stats['teacher_talk_pct']}%",   f"{stats['teacher_talk_time_s']}s")
    metric_card(c3, "Student Talk",     f"{stats['student_talk_pct']}%",   f"{stats['student_talk_time_s']}s")
    metric_card(c4, "Teacher Qs",       str(stats["teacher_questions"]),   "detected")
    metric_card(c5, "Student Responses",str(stats["student_responses"]),   f"+{stats['student_questions']} student Qs")

    badge_cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[level]
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;margin:0.5rem 0 0;">
      <span class="engagement-badge {badge_cls}">{level} Engagement</span>
      <span style="color:#4a5568;font-size:0.8rem;">
        Language: <strong style="color:#718096">{transcript.get('language','?').upper()}</strong>
        &nbsp;·&nbsp; Model: <strong style="color:#718096">{transcript.get('model','—')}</strong>
        &nbsp;·&nbsp; Teacher: <strong style="color:#718096">{transcript.get('teacher_id','—')}</strong>
      </span>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="summary-box">💡 {analysis["summary"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Engagement Metrics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = go.Figure(go.Pie(
            labels=["Teacher", "Student", "Silence"],
            values=[stats["teacher_talk_time_s"], stats["student_talk_time_s"], stats["silence_time_s"]],
            hole=0.6, marker_colors=["#63b3ed","#68d391","#4a5568"],
            textinfo="label+percent", textfont=dict(color="#e2e8f0", size=11),
        ))
        fig.update_layout(
            title=dict(text="Talk Time Distribution", font=dict(color="#a0aec0", size=13)),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            margin=dict(t=40,b=10,l=10,r=10), font=dict(family="Sora"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        m_names = ["TDR", "SPI", "QRR"]
        m_vals  = [metrics["teacher_dominance_ratio"],
                   metrics["student_participation_indicator"],
                   metrics["question_response_rate"]]
        ideal   = [(0.55, 0.70), (0.30, 0.50), (0.70, 1.00)]
        colors  = [
            "#68d391" if lo <= v <= hi else "#f6ad55" if abs(v-(lo+hi)/2) < 0.25 else "#fc8181"
            for v,(lo,hi) in zip(m_vals, ideal)
        ]
        fig2 = go.Figure(go.Bar(
            x=m_names, y=m_vals, marker_color=colors,
            text=[f"{v:.2f}" for v in m_vals], textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig2.update_layout(
            title=dict(text="Engagement Metrics (0–1)", font=dict(color="#a0aec0", size=13)),
            yaxis=dict(range=[0,1.15], gridcolor="#1a202c", color="#4a5568"),
            xaxis=dict(color="#718096"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40,b=20,l=10,r=10), font=dict(family="Sora"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = go.Figure()
        cmap = {"TEACHER": "#63b3ed", "STUDENT": "#68d391"}
        for s in segs:
            role = s.get("role", "STUDENT")
            fig3.add_shape(type="rect",
                x0=s["start"], x1=s["end"],
                y0=0, y1=1 if role == "TEACHER" else 0.5,
                fillcolor=cmap.get(role, "#718096"), opacity=0.7, line_width=0)
        fig3.update_layout(
            title=dict(text="Speaker Timeline", font=dict(color="#a0aec0", size=13)),
            xaxis=dict(title="Time (s)", color="#4a5568", gridcolor="#1a202c"),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40,b=30,l=10,r=10), height=200,
            showlegend=False, font=dict(family="Sora"),
        )
        fig3.add_annotation(x=0,   y=1.15, text="■ Teacher", showarrow=False, font=dict(color="#63b3ed", size=10), xref="paper", yref="paper")
        fig3.add_annotation(x=0.5, y=1.15, text="■ Student", showarrow=False, font=dict(color="#68d391", size=10), xref="paper", yref="paper")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-hdr">Metric Explanations</div>', unsafe_allow_html=True)
    METRIC_INFO = {
        "Teacher Dominance Ratio (TDR)": {
            "formula": "Teacher Talk Time ÷ Total Spoken Time",
            "ideal": "0.55 – 0.70", "value": metrics["teacher_dominance_ratio"],
            "desc": "High TDR (>0.75) = lecture-heavy. Low = student-led or balanced.",
        },
        "Student Participation Indicator (SPI)": {
            "formula": "(Student Turns ÷ Total Turns) + 0.05 × Student Questions",
            "ideal": "0.30 – 0.50", "value": metrics["student_participation_indicator"],
            "desc": "Turn frequency + bonus for student-initiated questions.",
        },
        "Interaction Density (ID)": {
            "formula": "Speaker Alternations ÷ Duration (minutes)",
            "ideal": "> 5 / min", "value": metrics["interaction_density"],
            "desc": "How often teacher↔student switches per minute. Higher = more dialogue.",
        },
        "Question Response Rate (QRR)": {
            "formula": "Student Responses ÷ Teacher Questions (capped 1.0)",
            "ideal": "> 0.70", "value": metrics["question_response_rate"],
            "desc": "Fraction of teacher questions that received a student response.",
        },
    }
    cols = st.columns(2)
    for i, (name, info) in enumerate(METRIC_INFO.items()):
        with cols[i % 2]:
            v = info["value"]
            display = f"{v:.2f}" if isinstance(v, float) else str(v)
            st.markdown(f"""
            <div class="metric-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div class="metric-label">{name}</div>
                <div class="metric-value">{display}</div>
              </div>
              <div style="margin:0.4rem 0;font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                   color:#4a90c4;background:#0a1e35;padding:0.3rem 0.6rem;border-radius:4px;">
                {info['formula']}
              </div>
              <div style="color:#4a5568;font-size:0.76rem;">Ideal: <span style="color:#718096">{info['ideal']}</span></div>
              <div style="color:#a0aec0;font-size:0.8rem;margin-top:0.2rem;">{info['desc']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Labeled Transcript</div>', unsafe_allow_html=True)
    fc1, fc2, _ = st.columns([1, 1, 3])
    show_teacher = fc1.checkbox("Teacher", value=True)
    show_student = fc2.checkbox("Student", value=True)

    html = ""
    for seg in segs:
        role = seg.get("role", "STUDENT")
        if role == "TEACHER" and not show_teacher: continue
        if role == "STUDENT" and not show_student: continue
        cls  = "seg-teacher" if role == "TEACHER" else "seg-student"
        icon = "👩‍🏫" if role == "TEACHER" else "🙋"
        spk  = seg.get("speaker_id", "")
        q    = " ❓" if seg.get("is_question") else ""
        html += f"""
        <div class="{cls}">
          <span class="seg-meta">{seg['start']:.1f}s–{seg['end']:.1f}s</span>
          <strong>{icon} {role} ({spk})</strong>{q}: {seg['text']}
        </div>"""

    st.markdown(
        f'<div style="max-height:480px;overflow-y:auto;background:#0a0f1a;border-radius:10px;padding:0.75rem;">{html}</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-hdr">Export</div>', unsafe_allow_html=True)
    import json
    ec1, ec2 = st.columns(2)
    with ec1:
        df = pd.DataFrame([
            {"start": s["start"], "end": s["end"], "role": s.get("role"),
             "speaker_id": s.get("speaker_id"), "is_question": s["is_question"], "text": s["text"]}
            for s in segs
        ])
        st.download_button("⬇️ Transcript CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="transcript.csv", mime="text/csv")
    with ec2:
        out = {"transcript": transcript, "analysis": {
            "stats": stats, "metrics": metrics,
            "engagement_level": level, "summary": analysis["summary"],
        }}
        st.download_button("⬇️ Full JSON",
            json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="classroom_analysis.json", mime="application/json")

    if analysis["silences"]:
        with st.expander(f"🔇 Silence Gaps ({len(analysis['silences'])})"):
            st.dataframe(pd.DataFrame(analysis["silences"]), use_container_width=True, hide_index=True)

else:
    if not uploaded_file:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#4a5568;">
          <div style="font-size:3rem;">🎙️</div>
          <div style="font-size:1.1rem;color:#718096;margin-top:1rem;">Upload an audio file to begin.</div>
          <div style="font-size:0.82rem;margin-top:0.5rem;color:#4a5568;">
            Add your HF token in the sidebar · Models cache after first load
          </div>
        </div>""", unsafe_allow_html=True)