# text_generator.py
from functools import lru_cache
from typing import List, Union

import torch
from transformers import pipeline

DEFAULT_MODEL = "google/flan-t5-base"  # instruction-tuned, CPU-friendly
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=4)
def load_generator(model_name: str = DEFAULT_MODEL, device: str = DEVICE):
    """
    Load the generation pipeline: text2text for FLAN/T5, text-generation otherwise.
    """
    device_index = 0 if device == "cuda" else -1
    is_t5 = ("t5" in model_name.lower()) or ("flan" in model_name.lower())
    task = "text2text-generation" if is_t5 else "text-generation"

    return pipeline(
        task,
        model=model_name,
        tokenizer=model_name,
        device=device_index,
        model_kwargs={"low_cpu_mem_usage": True},
    )

def build_generation_prompt(
    user_prompt: str,
    emotions: Union[str, List[str]],
    target_words: int = 200,
    mode: str = "paragraph",
):
    if isinstance(emotions, (list, tuple)):
        emotions_list = list(emotions)[:2]
    else:
        emotions_list = [str(emotions)] if emotions else ["neutral"]

    #Emotions maping   
    STYLE_MAP = {
        "caring": "warm and empathetic",
        "relief": "reassured and calm",
        "optimism": "hopeful and forward-looking",
        "confidence": "steady and confident",
        "joy": "positive and encouraging",
        "sadness": "gentle and supportive",
        "fear": "calm and reassuring",
        "frustration": "understanding and constructive",
        "confusion": "clear and patient",
        "neutral": "balanced and respectful",
    }
    tone = ", ".join(STYLE_MAP.get(e, e) for e in emotions_list)

    q = user_prompt.strip()
    opener = "Here are a few things that will help:"

    if mode == "essay":
        form_rule = (
            "Write 3-5 short paragraphs (each 2-4 sentences). "
            "Total length near the target. Avoid filler and thank-yous."
        )
    else:
        form_rule = (
            "Write 4-9 concise sentences in ONE paragraph. "
            "Avoid filler and thank-yous."
        )

    # Two-shot to anchor style and stop echoing
    return (
        f"You are a practical mentor. Tone: {tone}.\n"
        f"{form_rule}\n"
        "Do NOT restate or paraphrase the question. Do NOT ask the user to rephrase it.\n"
        f"Always start your answer with: \"{opener}\"\n"
        "Prefer specific, actionable steps. Keep each sentence under ~18 words. No generic phrases like 'build your model'.\n"
        "\n"
        "Example 1 (caring):\n"
        "Question: I'm overwhelmed learning data structures. Where do I even start?\n"
        f"Answer: {opener} Pick one curated course and finish arrays → linked lists → stacks/queues in order. "
        "Study 25 minutes daily and re-code each example from memory. Solve 2 easy problems per topic on one site. "
        "Keep one-page notes with a minimal snippet per structure. Review weekly by building a tiny project using two structures.\n"
        "\n"
        "Example 2 (optimistic, confident):\n"
        "Question: I keep getting stuck debugging simple bugs and lose motivation.\n"
        f"Answer: {opener} Reproduce the bug in a 10-20 line snippet and add one assert. "
        "Use a step debugger or prints to frame the first wrong value. Write a failing test, fix it, and keep the test. "
        "Log each fix in a short bug diary. Celebrate one small win per day to rebuild momentum.\n"
        "\n"
        "Now answer the user:\n"
        f"Question: {q}\n"
        "Answer:"
    )


def generate_text(
    user_prompt: str,
    emotions: Union[str, List[str]] = "",
    model_name: str = DEFAULT_MODEL,
    target_words: int = 200,
    mode: str = "paragraph",
    temperature: float = 0.5,
    top_k: int = 50,
    device: str = DEVICE,
) -> str:
    gen = load_generator(model_name, device)
    prompt = build_generation_prompt(user_prompt, emotions, target_words, mode)
    is_t5 = ("t5" in model_name.lower()) or ("flan" in model_name.lower())

    # Words → tokens (≈1.6 tokens/word). Hard caps to reduce rambling.
    tokens_per_word = 1.6
    if mode == "essay":
        max_new = int(min(900, max(240, target_words * tokens_per_word * 2.2)))
        min_new = int(min(600, max(90, target_words * 0.7)))
    else:
        max_new = int(min(320, max(120, target_words * tokens_per_word * 1.2)))
        min_new = int(min(160, max(55, target_words * 0.5)))

    if is_t5:
        gen_args = dict(
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            num_beams=4,
            do_sample=False,                 # deterministic first pass
            no_repeat_ngram_size=4,          # stricter n-gram block
            encoder_no_repeat_ngram_size=3,  # limit copying from input
            repetition_penalty=1.15,         # discourage loops
            length_penalty=0.9,
            eos_token_id=getattr(gen.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(gen.tokenizer, "pad_token_id", None)
                        or getattr(gen.tokenizer, "eos_token_id", None),
        )
    else:
        gen_args = dict(
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            do_sample=True,
            temperature=min(temperature, 0.9),
            top_k=top_k,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            return_full_text=False,
            eos_token_id=getattr(gen.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(gen.tokenizer, "pad_token_id", None)
                        or getattr(gen.tokenizer, "eos_token_id", None),
        )

    out = gen(prompt, **gen_args)
    text = out[0]["generated_text"].strip()

    # Fallback once if too short or echo-ish (sampling for variety)
    topic_only = user_prompt.strip().lower()
    if (len(text.split()) < 25) or (topic_only in text.lower() and len(text.split()) < 35):
        if is_t5:
            out = gen(
                prompt,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                do_sample=True,
                top_p=0.9,
                temperature=min(temperature, 0.7),
                no_repeat_ngram_size=4,
                repetition_penalty=1.15,
                eos_token_id=getattr(gen.tokenizer, "eos_token_id", None),
                pad_token_id=getattr(gen.tokenizer, "pad_token_id", None)
                            or getattr(gen.tokenizer, "eos_token_id", None),
            )
            text = out[0]["generated_text"].strip()

    return text

