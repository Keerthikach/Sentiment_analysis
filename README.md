Emotion-Aware Text Generator (GoEmotions + FLAN-T5)

Generate a paragraph or short essay that matches the sentiment/emotions inferred from a user’s prompt.
Built with Streamlit, Hugging Face Transformers, GoEmotions (for emotion detection), and FLAN-T5-base (for generation).

Project Explanation

This app takes a user prompt (e.g., “I’m stuck and can’t learn to code—any tips?”), detects the underlying emotions using a GoEmotions-based classifier, maps those emotions to a tone (e.g., caring → warm & empathetic), and then prompts an instruction-tuned generator (FLAN-T5-base) to write a response in that tone.

The UI lets you adjust length (paragraph vs essay), target words, temperature, and override emotions manually.

Methodology
1) Problem framing

We want to produce text that aligns with the detected sentiment/emotions of the input prompt.

2) Pipeline

Input → User types a prompt in Streamlit.

Emotion detection → GoEmotions classifier returns top labels (multi-label with threshold).

Tone mapping → Map 1–2 labels to short tone descriptors (e.g., relief → reassured and calm).

Controlled prompting → Build a two-shot instruction prompt that:

forbids restating the question,

starts with a fixed opener: “Here are a few things that will help:”,

encourages specific, actionable tips,

incorporates the tone from step 3.

Decoding → FLAN-T5-base with:

num_beams=4 (deterministic first pass),

min_new_tokens (avoid early stop),

no_repeat_ngram_size and encoder_no_repeat_ngram_size (limit echo/repetition),

fallback: single sampling pass (low temp, top-p) only if result is too short or echo-ish.

Length control → Target words & mode (Paragraph/Essay) scale token budgets.

History → Prompts, settings, and outputs saved to CSV.

3) Emotion → Tone mapping

Only the top 1–2 emotions are used to keep style crisp. Example mappings:

caring → warm and empathetic

relief → reassured and calm

optimism → hopeful and forward-looking

confusion → clear and patient

neutral → balanced and respectful

4) Prompting strategy (anti-echo)

Two concise examples (two-shot) teach the format.

Fixed opener anchors the first sentence.

Explicit “Do NOT restate the question.”

For Essay, request 3–5 short paragraphs; for Paragraph, 4–6 concise sentences.

5) Decoding strategy

Primary pass: beam search + length & repetition constraints.

Fallback: one sampling pass (low temperature/top-p) only if needed.

Datasets / Models Used
Emotion/Sentiment

Classifier: joeddav/distilbert-base-uncased-go-emotions-student
Distilled from GoEmotions (≈58k Reddit comments, 27 emotions + neutral).
We run in multi-label mode with a threshold; if nothing passes, we keep the top label.

Text Generation

Generator: google/flan-t5-base (~250M params)
Instruction-tuned, CPU-friendly, decent adherence to directions at small scale.

I did not use the Ekman ^ emotions because it is pre-trained only on a 6 emotions but with GoEmotions it has a wider range of emotions available

Challenges

Tried many different Models and later chose Flan-5 to be teh based, The best text generation models had too many parameters and hence required more space and ran slower. The ones which had less parametres never had proper context and hallucinated, Hence I chose Flan-5 which had teh best of both worlds.

Hardware limits (no GPU, limited RAM)

Large LLMs are too heavy.

Solution: FLAN-T5-base for generation + distilled GoEmotions for classification.

Echoing / generic rambling

Small models repeat the input or ramble.

Solution: Two-shot prompt, fixed opener, beam search + min_new_tokens, no_repeat_ngram_size, encoder_no_repeat_ngram_size, and a single low-temp sampling fallback.

Keeping outputs concise & on-topic

Long free-form requests cause drift.

Solution: Target words & mode (Paragraph/Essay) with hard token caps; prompt emphasizes specific, actionable advice.

Emotion coverage & realism

Too many labels = muddy tone; too few = rigid.

Solution: Limit to top 1–2 labels and allow manual override.

Setup & Run Instructions
1) Prerequisites

Python 3.10+ recommended

A machine with internet access for the initial model downloads

(Optional) A virtual environment (strongly recommended)

2) Create & activate a virtual environment
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1


3) Project files

Place these files in the project root:

app.py
components.py
config.py
emotion_detector.py
history.py
text_generator.py
requirements.txt

4) Install dependencies

Create a requirements.txt like:

pip install -r requirements.txt


Tip (Torch CPU wheel):
If pip install torch fails or chooses a GPU build, install the appropriate CPU wheel from PyPI for your OS/Python.

5) Run the app
streamlit run app.py


Open the local URL shown in your terminal (typically http://localhost:8501).

**This may take a while as the model and generator need to be downloaded/loaded from Huggingface hub and the text generation takes a little bit of time**

# The model still rambles and hallucinates a little as it s a small model with just 250M parameteres. this model was chosen to be the best model which puts teh least amount of pressure on any system. Even ones without a GPU or a smaller RAM.

Usage

Type your prompt (e.g., “I’m stuck and not progressing in Python—what should I do?”).

The app detects emotions and shows them (with scores).

(Optional) Toggle Manual override to pick emotions yourself.

Choose Mode (Paragraph or Essay) and Target length (words).

Click Generate.

View the output; download history as CSV if desired.

Tips for better outputs

Ask for specific help (e.g., “stuck on loops and functions”) rather than very broad queries.

For tighter responses, keep Paragraph mode and moderate Target words (120–250).

Use a lower temperature (≤0.7) if the fallback triggers often.

Configuration

Edit defaults in config.py:

GOEMOTIONS_MODEL – emotion classifier checkpoint

GENERATION_MODEL – reference name (generator uses its own DEFAULT_MODEL)

HF_DEVICE – -1 CPU, or 0 for GPU if available

MAX_WORDS_DEFAULT, TEMPERATURE_DEFAULT – UI defaults

MULTILABEL_THRESHOLD_DEFAULT – emotion pick threshold

GOEMOTIONS_LABELS – control sidebar ordering

Troubleshooting

It repeats the question / rambles
Use Paragraph mode, reduce Target words, keep temp ≤ 0.7. The app already enforces beam search + anti-echo constraints; extreme prompts can still confuse tiny models.

Slow / model too big
First run downloads models. Subsequent runs are much faster. If still slow, lower Target words or switch from Essay to Paragraph.

Torch install issues
Ensure you installed a suitable CPU wheel for your OS/Python. Check PyPI if default pip install torch fails.

License / Attribution

Emotion model: joeddav/distilbert-base-uncased-go-emotions-student (distilled from Google’s GoEmotions).

Generator model: google/flan-t5-base.

Built with Hugging Face Transformers & Streamlit.