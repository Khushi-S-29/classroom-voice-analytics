import os, sys, json, tempfile
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from transcriber import DEMO_TRANSCRIPT
from analyzer import analyze_transcript

st.set_page_config(
    page_title="Classroom Analytics",
    page_icon="🎙️", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.main { background: #0d1117; }
.block-container { padding: 1.5rem 2rem; }
.mg-header {
    background: linear-gradient(135deg, #1a2744 0%, #0f1923 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    border: 1px solid rgba(99,179,237,0.2);
}
.mg-header h1 { color: #e2e8f0; font-size: 2rem; font-weight: 700; margin: 0 0 0.25rem; }
.mg-header p  { color: #718096; font-size: 0.95rem; margin: 0; }
.mg-logo { font-size: 2.2rem; margin-right: 1rem; }
.metric-card {
    background: #161d2d; border-radius: 12px; padding: 1.2rem 1.4rem;
    border: 1px solid rgba(99,179,237,0.15); margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(99,179,237,0.5); }
.metric-label { color: #718096; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { color: #e2e8f0; font-size: 1.9rem; font-weight: 700; line-height: 1.1; }
.metric-sub   { color: #4a5568; font-size: 0.78rem; margin-top: 0.2rem; }
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
.seg-teacher { background:#0a1e35; border-left:3px solid #63b3ed; }
.seg-student { background:#0d2b1a; border-left:3px solid #68d391; }
.seg-q       { background:#1a1a0d; border-left:3px solid #f6e05e; }
.seg { padding:0.5rem 0.9rem; border-radius:0 8px 8px 0; margin:0.25rem 0; font-size:0.88rem; }
.seg-meta { color:#4a5568; font-size:0.72rem; font-family:'JetBrains Mono',monospace; }
.seg-text { color:#e2e8f0; }
.seg-tag  { font-size:0.7rem; font-weight:600; padding:0.1rem 0.4rem; border-radius:4px; margin-right:0.3rem; }
.tag-t { background:#1a3a5c; color:#90cdf4; }
.tag-s { background:#1a3d2b; color:#9ae6b4; }
.tag-q { background:#2d2a0d; color:#faf089; }
.section-hdr {
    color:#4a5568; font-size:0.7rem; text-transform:uppercase;
    letter-spacing:0.1em; font-weight:600;
    margin:1.5rem 0 0.75rem; padding-bottom:0.4rem; border-bottom:1px solid #1a202c;
}
section[data-testid="stSidebar"] { background: #0a0f1a; }
section[data-testid="stSidebar"] .block-container { padding: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mg-header">
  <div style="display:flex;align-items:center;">
    <span class="mg-logo">🏫</span>
    <div>
      <h1>MakerGhat Classroom Analytics</h1>
      <p>Audio → Insights · Indic Language Support · Offline-First MVP</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Input Mode")
    input_mode = st.radio(
        "Choose input",
        ["Upload Audio", "Load Transcript JSON", "Demo Mode (Hindi)"],
        index=2,
    )
    st.divider()

    if input_mode == "🎤 Upload Audio (Whisper)":
        st.markdown("**Whisper Settings**")

        try:
            import faster_whisper
            backend_label = "faster-whisper ⚡ (fast)"
        except ImportError:
            try:
                import whisper
                backend_label = "openai-whisper (standard)"
            except ImportError:
                backend_label = "❌ No backend — install one below"

        st.caption(f"Backend: **{backend_label}**")
        if "faster-whisper" not in backend_label and "openai-whisper" not in backend_label:
            st.code("pip install faster-whisper", language="bash")

        whisper_model = st.selectbox(
            "Model size",
            ["tiny", "base", "small", "medium"],
            index=0,
            help="tiny=fastest · small=best Hindi accuracy · medium=best overall",
        )
        lang_choice = st.selectbox(
            "Language hint",
            ["Auto-detect", "Hindi (hi)", "Marathi (mr)", "Tamil (ta)",
             "Bengali (bn)", "Telugu (te)", "Gujarati (gu)", "English (en)"],
        )
        lang_map = {
            "Auto-detect": None, "Hindi (hi)": "hi", "Marathi (mr)": "mr",
            "Tamil (ta)": "ta", "Bengali (bn)": "bn", "Telugu (te)": "te",
            "Gujarati (gu)": "gu", "English (en)": "en",
        }
        lang_code = lang_map[lang_choice]

        speed_est = {"tiny": "~4–15s", "base": "~10–35s", "small": "~25–90s", "medium": "~60–180s"}
        st.caption(f"Est. time for 5-min audio: **{speed_est[whisper_model]}**")

        uploaded_file = st.file_uploader(
            "Upload audio", type=["wav", "mp3", "m4a", "ogg", "flac"],
        )

    st.divider()
    st.markdown("""
    <div style="color:#4a5568;font-size:0.75rem;line-height:1.6;">
    <strong style="color:#718096">Metrics</strong><br>
    • Teacher Dominance Ratio (TDR)<br>
    • Student Participation Indicator (SPI)<br>
    • Interaction Density (alternations/min)<br>
    • Question Response Rate (QRR)
    </div>
    """, unsafe_allow_html=True)

transcript = analysis = error_msg = None

if input_mode == "Demo Mode (Hindi)":
    transcript = DEMO_TRANSCRIPT
    analysis   = analyze_transcript(transcript)

elif input_mode == "Load Transcript JSON":
    jf = st.file_uploader("Upload transcript JSON", type=["json"])
    if jf:
        try:
            transcript = json.load(jf)
            analysis   = analyze_transcript(transcript)
        except Exception as e:
            error_msg = f"Failed to load JSON: {e}"

elif input_mode == "🎤 Upload Audio (Whisper)":
    if uploaded_file:
        with st.spinner("Transcribing… (thanks for your patience)"):
            try:
                from transcriber import transcribe_audio
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                transcript = transcribe_audio(tmp_path, model_size=whisper_model, language=lang_code)
                os.unlink(tmp_path)
                analysis   = analyze_transcript(transcript)
                st.success(f"Done — detected language: **{transcript.get('language','?').upper()}**")
            except ImportError as e:
                error_msg = str(e)
            except Exception as e:
                error_msg = f"Transcription failed: {e}"
    else:
        st.info("📁 Upload an audio file to begin.")

if error_msg:
    st.error(error_msg)

if analysis:
    stats   = analysis["stats"]
    metrics = analysis["metrics"]
    segs    = analysis["labeled_segments"]

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
    metric_card(c1, "Duration",          f"{dur_min}m {dur_sec}s", f"{stats['total_segments']} segments")
    metric_card(c2, "Teacher Talk",      f"{stats['teacher_talk_pct']}%", f"{stats['teacher_talk_time_s']}s")
    metric_card(c3, "Student Talk",      f"{stats['student_talk_pct']}%", f"{stats['student_talk_time_s']}s")
    metric_card(c4, "Teacher Questions", str(stats["teacher_questions"]), "detected")
    metric_card(c5, "Student Responses", str(stats["student_responses"]), f"+{stats['student_questions']} student Qs")

    level     = analysis["engagement_level"]
    badge_cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}[level]
    lang_disp = transcript.get("language", "?").upper()
    model_disp = transcript.get("model", "—")
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;margin:0.5rem 0;">
      <span class="engagement-badge {badge_cls}">{level} Engagement</span>
      <span style="color:#4a5568;font-size:0.8rem;">Language: <strong style="color:#718096">{lang_disp}</strong></span>
      <span style="color:#4a5568;font-size:0.8rem;">Model: <strong style="color:#718096">{model_disp}</strong></span>
      <span style="color:#4a5568;font-size:0.8rem;">File: <strong style="color:#718096">{transcript.get('audio_file','—')}</strong></span>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="summary-box">💡 {analysis["summary"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Engagement Metrics</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        fig = go.Figure(go.Pie(
            labels=["Teacher", "Student", "Silence"],
            values=[stats["teacher_talk_time_s"], stats["student_talk_time_s"], stats["silence_time_s"]],
            hole=0.6, marker_colors=["#63b3ed", "#68d391", "#4a5568"],
            textinfo="label+percent", textfont=dict(color="#e2e8f0", size=11),
        ))
        fig.update_layout(
            title=dict(text="Talk Time Distribution", font=dict(color="#a0aec0", size=13)),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            margin=dict(t=40, b=10, l=10, r=10), font=dict(family="Sora"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        m_names = ["TDR", "SPI", "QRR"]
        m_vals  = [metrics["teacher_dominance_ratio"],
                   metrics["student_participation_indicator"],
                   metrics["question_response_rate"]]
        ideal   = [(0.55, 0.70), (0.30, 0.50), (0.70, 1.00)]
        colors  = [
            "#68d391" if lo <= v <= hi else "#f6ad55" if abs(v - (lo+hi)/2) < 0.25 else "#fc8181"
            for v, (lo, hi) in zip(m_vals, ideal)
        ]
        fig2 = go.Figure(go.Bar(
            x=m_names, y=m_vals, marker_color=colors,
            text=[f"{v:.2f}" for v in m_vals], textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig2.update_layout(
            title=dict(text="Engagement Metrics (0–1)", font=dict(color="#a0aec0", size=13)),
            yaxis=dict(range=[0, 1.15], gridcolor="#1a202c", color="#4a5568"),
            xaxis=dict(color="#718096"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=10, r=10), font=dict(family="Sora"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = go.Figure()
        color_map = {"TEACHER": "#63b3ed", "STUDENT": "#68d391"}
        for s in segs:
            fig3.add_shape(
                type="rect",
                x0=s["start"], x1=s["end"],
                y0=0, y1=1 if s["speaker"] == "TEACHER" else 0.5,
                fillcolor=color_map[s["speaker"]], opacity=0.7, line_width=0,
            )
        fig3.update_layout(
            title=dict(text="Speaker Timeline", font=dict(color="#a0aec0", size=13)),
            xaxis=dict(title="Time (s)", color="#4a5568", gridcolor="#1a202c"),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=30, l=10, r=10), height=200,
            showlegend=False, font=dict(family="Sora"),
        )
        fig3.add_annotation(x=0,   y=1.15, text="■ Teacher", showarrow=False,
                            font=dict(color="#63b3ed", size=10), xref="paper", yref="paper")
        fig3.add_annotation(x=0.5, y=1.15, text="■ Student", showarrow=False,
                            font=dict(color="#68d391", size=10), xref="paper", yref="paper")
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
            "desc": "Captures turn frequency + bonus for student-initiated questions.",
        },
        "Interaction Density (ID)": {
            "formula": "Speaker Alternations ÷ Duration (minutes)",
            "ideal": "> 5 / min", "value": metrics["interaction_density"],
            "desc": "How often teacher↔student switches happen per minute. Higher = more dialogue.",
        },
        "Question Response Rate (QRR)": {
            "formula": "Student Responses ÷ Teacher Questions (capped 1.0)",
            "ideal": "> 0.70", "value": metrics["question_response_rate"],
            "desc": "Fraction of teacher questions that got a student response.",
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
              <div style="margin:0.4rem 0;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#4a90c4;background:#0a1e35;padding:0.3rem 0.6rem;border-radius:4px;">
                {info['formula']}
              </div>
              <div style="color:#4a5568;font-size:0.78rem;margin-bottom:0.2rem;">Ideal: <span style="color:#718096">{info['ideal']}</span></div>
              <div style="color:#a0aec0;font-size:0.8rem;">{info['desc']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Labeled Transcript</div>', unsafe_allow_html=True)
    fc1, fc2, _ = st.columns([1, 1, 3])
    show_teacher = fc1.checkbox("Show Teacher", value=True)
    show_student = fc2.checkbox("Show Student", value=True)

    html = ""
    for seg in segs:
        if seg["speaker"] == "TEACHER" and not show_teacher: continue
        if seg["speaker"] == "STUDENT" and not show_student: continue
        is_q = seg.get("is_question", False)
        if is_q:
            cls, tag = "seg seg-q", '<span class="seg-tag tag-q">❓ Q</span>'
        elif seg["speaker"] == "TEACHER":
            cls, tag = "seg seg-teacher", '<span class="seg-tag tag-t">👩‍🏫 T</span>'
        else:
            cls, tag = "seg seg-student", '<span class="seg-tag tag-s">🙋 S</span>'
        html += f"""
        <div class="{cls}">
          <span class="seg-meta">{seg['start']:.1f}s – {seg['end']:.1f}s</span>
          {tag}<span class="seg-text">{seg['text']}</span>
        </div>"""
    st.markdown(
        f'<div style="max-height:420px;overflow-y:auto;background:#0a0f1a;border-radius:10px;padding:0.75rem;">{html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-hdr">Export</div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)
    with ec1:
        df = pd.DataFrame([
            {"start": s["start"], "end": s["end"], "speaker": s["speaker"],
             "is_question": s["is_question"], "text": s["text"]}
            for s in segs
        ])
        st.download_button("⬇️ Download Transcript CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="transcript.csv", mime="text/csv")
    with ec2:
        out = {"transcript": transcript, "analysis": {
            "stats": stats, "metrics": metrics,
            "engagement_level": analysis["engagement_level"],
            "summary": analysis["summary"],
        }}
        st.download_button("⬇️ Download Full JSON",
            json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="classroom_analysis.json", mime="application/json")

    if analysis["silences"]:
        with st.expander(f"🔇 Silence Gaps ({len(analysis['silences'])})"):
            st.dataframe(pd.DataFrame(analysis["silences"]),
                         use_container_width=True, hide_index=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#4a5568;">
      <div style="font-size:3rem;">🎙️</div>
      <div style="font-size:1.1rem;color:#718096;margin-top:1rem;">
        Select an input mode from the sidebar to get started.
      </div>
      <div style="font-size:0.85rem;margin-top:0.5rem;">
        Try <strong style="color:#63b3ed">Demo Mode</strong> for an instant Hindi classroom example.
      </div>
    </div>""", unsafe_allow_html=True)