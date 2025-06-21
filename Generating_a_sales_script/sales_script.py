import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from langdetect import detect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Summarization model (multilingual capable)
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"  # English; for Russian you can use mt5-base or similar
# Generation model for explanation (multilingual)
GENERATION_MODEL = "bigscience/bloomz-560m"


def load_models():
    print("Loading models...")
    summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL).to(device)

    generator_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    generator_model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL).to(device)

    return (summarizer_tokenizer, summarizer_model, generator_tokenizer, generator_model)


def summarize_text(text, tokenizer, model, max_length=150):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def detect_language(text):
    try:
        lang = detect(text)
        print(f"🌐 Detected language: {lang}")
        return lang
    except Exception:
        return "en"


def chunk_text(text, max_length=800):
    # Naive chunker by sentences limited to approx max_length chars
    import re

    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) < max_length:
            current_chunk += sent + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def generate_explanation(
    greeting, needs, pitch, closing, lang, tokenizer, model, max_tokens=256
):
    prompt = f"""
You are an expert sales coach.

Given the following sales call structure:

Greeting: {greeting}
Needs Discovery: {needs}
Pitch: {pitch}
Closing: {closing}

Explain:
1. Why this structure is effective.
2. The psychological triggers used in each stage.
3. Which phrases work best and why.

Provide your answer in clear, concise bullet points.
Answer in {lang}.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated.strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python sales_script_generator.py <transcript.txt>")
        return

    input_path = sys.argv[1]
    with open(input_path, "r", encoding="utf-8") as f:
        transcript = f.read().strip()

    lang = detect_language(transcript)

    # Load models
    summarizer_tokenizer, summarizer_model, generator_tokenizer, generator_model = load_models()

    # Chunk transcript into approx 4 parts representing sales stages
    chunks = chunk_text(transcript, max_length=800)
    if len(chunks) < 4:
        # pad with empty strings if fewer chunks
        chunks += [""] * (4 - len(chunks))

    # Assign chunks to stages
    greeting_raw = chunks[0]
    needs_raw = chunks[1]
    pitch_raw = chunks[2]
    closing_raw = chunks[3]

    # Summarize each stage (optional but recommended)
    greeting = summarize_text(greeting_raw, summarizer_tokenizer, summarizer_model)
    needs = summarize_text(needs_raw, summarizer_tokenizer, summarizer_model)
    pitch = summarize_text(pitch_raw, summarizer_tokenizer, summarizer_model)
    closing = summarize_text(closing_raw, summarizer_tokenizer, summarizer_model)

    print("\n--- Generated Sales Script Template ---")
    print(f"Greeting:\n{greeting}\n")
    print(f"Needs Discovery:\n{needs}\n")
    print(f"Pitch:\n{pitch}\n")
    print(f"Closing:\n{closing}\n")

    # Generate explanation
    explanation = generate_explanation(
        greeting, needs, pitch, closing, lang, generator_tokenizer, generator_model
    )

    print("\n--- Sales Script Explanation ---")
    print(explanation)


if __name__ == "__main__":
    main()
