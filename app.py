
import streamlit as st
import emotion_detector
import text_generator
import components
import history
from config import (
    HF_DEVICE, MULTILABEL_THRESHOLD_DEFAULT, GOEMOTIONS_LABELS,
    MAX_WORDS_DEFAULT
)
from datetime import datetime

# Streamlit page config (first Streamlit call)
st.set_page_config(page_title="Sentiment and Emotions Analyser", layout="wide")

# Initialize generator in session state
if "generator" not in st.session_state:
    st.session_state.generator = text_generator.load_generator(device=HF_DEVICE)

# Initialize emotion detection pipeline
if "emotion_pipe" not in st.session_state:
    st.session_state.emotion_pipe = emotion_detector.load_goemotions_pipeline(device=HF_DEVICE)

st.title("Sentiment and Emotions Analyser")
st.markdown(
    "Paste a prompt. This bot detects emotions and returns short texts based on the prompt"
    ". You can also manually choose emotions (single/multi)."
)

# Sidebar controls
with st.sidebar:
    st.header("Generation Controls")
    mode = st.radio("Mode", ["Paragraph", "Essay"], horizontal=True, index=0)
    target_words = st.slider("Target length (words)", 60, 250, value=MAX_WORDS_DEFAULT)
    # Narrow temperature range: high temps (≥1.0) make tiny models ramble
    temperature = st.slider("Temperature", 0.1, 1.0, 0.5)
    st.markdown("---")

    st.markdown("### Detection")
    threshold = st.slider("Label threshold", 0.05, 0.7, MULTILABEL_THRESHOLD_DEFAULT)
    use_multilabel = st.checkbox("Use multi-label detection", value=True)
    st.markdown("---")

    if st.button("Reload model"):
        if "generator" in st.session_state:
            del st.session_state["generator"]
        st.session_state.generator = text_generator.load_generator(device=HF_DEVICE)
        st.success("Model reloaded. Generate again.")



# Main input
prompt = st.text_area("Enter your prompt / topic:", height=160)
generate_btn = st.button("Generate")

# Run detection if prompt present
detected_labels = ["neutral"]
detection_result = None
if prompt.strip():
    detection_result = emotion_detector.detect_emotions(prompt, threshold=threshold, device=HF_DEVICE)
    detected_labels = [lbl for lbl, _ in detection_result["predictions"]] \
                      if detection_result and detection_result.get("predictions") \
                      else ([detection_result["top_label"][0]] if detection_result and detection_result.get("top_label") else ["neutral"])

# Emotion selection UI
emotion_ui = components.emotion_sidebar_ui(detected_labels, options=GOEMOTIONS_LABELS)
selected_emotions = emotion_ui["selected"]

# Show detected labels
st.markdown("### Auto-detected emotions")
if detection_result and detection_result.get("predictions"):
    pred_text = ", ".join([f"{m} ({s:.2f})" for m, s in detection_result["predictions"]])
    st.write(pred_text)
else:
    st.write("No emotions detected (or prompt empty).")

# Generate on click
if generate_btn:
    if not prompt.strip():
        st.warning("Enter a prompt first.")
    else:
        used_emotions = selected_emotions if emotion_ui["manual"] else detected_labels

        with st.spinner("Generating..."):
            generated = text_generator.generate_text(
                user_prompt=prompt.strip(),
                emotions=used_emotions,
                target_words=target_words,
                mode=mode.lower(),            # "paragraph" | "essay"
                temperature=temperature,
                device=HF_DEVICE,
            )

        st.markdown("### Generated output")
        st.write(generated)

        # Save to history
        item = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": prompt,
            "used_emotions": "|".join(used_emotions),
            "generated": generated,
            "target_words": target_words,
            "temperature": temperature,
            "mode": mode,
        }

        history.save_history(item)

# History
st.markdown("---")
st.header("History")
df = history.load_history(limit=200)
components.show_history_table(df)
