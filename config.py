
# Central configuration

# Emotion model (GoEmotions distilled)
GOEMOTIONS_MODEL = "joeddav/distilbert-base-uncased-go-emotions-student"

# Generation model (reference; text_generator also sets DEFAULT_MODEL)
GENERATION_MODEL = "google/flan-t5-base"

# Defaults for generation controls
MAX_WORDS_DEFAULT = 200
TEMPERATURE_DEFAULT = 0.6

# Default device for HF pipelines: -1 for CPU, set to 0 for GPU
HF_DEVICE = -1

# Default threshold for selecting labels from GoEmotions (multi-label)
MULTILABEL_THRESHOLD_DEFAULT = 0.35

# GoEmotions label set for UI ordering (27 + neutral)
GOEMOTIONS_LABELS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion","curiosity",
    "desire","disappointment","disapproval","disgust","embarrassment","excitement","fear",
    "gratitude","grief","joy","love","nervousness","optimism","pride","realization","relief",
    "remorse","sadness","surprise","neutral"
]

# Persistence
HISTORY_FILE = "generation_history.csv"
