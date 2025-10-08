
import streamlit as st
from config import GOEMOTIONS_LABELS

#Manula emotions selection
def emotion_sidebar_ui(detected_labels, options=None):
    if options is None:
        options = GOEMOTIONS_LABELS
    st.sidebar.header("Emotion Controls")
    st.sidebar.markdown("Auto-detected emotions are shown. Toggle manual override to change them.")
    manual = st.sidebar.checkbox("Manual override", value=False)
    if manual:
        mode = st.sidebar.radio("Selection mode", ["single", "multi"], index=1)
        if mode == "single":
            chosen = st.sidebar.selectbox("Choose one emotion", options, index=options.index("neutral"))
            return {"manual": True, "mode": "single", "selected": [chosen]}
        default = detected_labels if detected_labels else ["neutral"]
        chosen_multi = st.sidebar.multiselect("Choose emotion(s)", options, default=default)
        return {"manual": True, "mode": "multi", "selected": chosen_multi}
    return {"manual": False, "mode": "auto", "selected": detected_labels or ["neutral"]}

#Shows history 
def show_history_table(df):
    if df.empty:
        st.info("No history yet.")
        return
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download history (CSV)", csv, file_name="generation_history.csv")
