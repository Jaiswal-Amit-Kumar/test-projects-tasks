import os
import sys
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from langdetect import detect
import re

# Models
SUM_MODEL = "sshleifer/distilbart-cnn-12-6"
GEN_MODEL = "bigscience/bloomz-560m"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Sales Script Analyzer", version="1.0")

# Load models once at startup
summ_tok = AutoTokenizer.from_pretrained(SUM_MODEL)
summ_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL).to(device)
gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(device)


def summarize_text(text, tokenizer, model, max_length=150):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    ids = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def chunk_text_by_stage(text):
    """
    More robust chunking for scripts explicitly labeled by stages:
    Looks for the 4 stages keywords and extracts their content.
    If missing any, fills with empty string.
    """
    stages = ["Greeting:", "Needs Discovery:", "Suggestion/Pitch:", "Closing:"]
    chunks = {}
    
    # Regex to split text by stages with content capture
    pattern = re.compile(r"(Greeting:|Needs Discovery:|Suggestion/Pitch:|Closing:)(.*?)(?=Greeting:|Needs Discovery:|Suggestion/Pitch:|Closing:|$)", re.S)
    matches = pattern.findall(text)

    for stage, content in matches:
        chunks[stage.rstrip(":")] = content.strip()

    # Ensure all stages are present, else empty
    return [
        chunks.get("Greeting", ""),
        chunks.get("Needs Discovery", ""),
        chunks.get("Suggestion/Pitch", ""),
        chunks.get("Closing", ""),
    ]


def generate_expl(greeting, needs, pitch, closing, lang):
    prompt = (
        "You are a highly experienced multilingual sales coach and communication expert.\n\n"
        "Analyze the following sales script, divided into four key stages:\n\n"
        f"1️⃣ Greeting:\n{greeting}\n\n"
        f"2️⃣ Needs Discovery:\n{needs}\n\n"
        f"3️⃣ Suggestion/Pitch:\n{pitch}\n\n"
        f"4️⃣ Closing:\n{closing}\n\n"
        "TASK A: Rewrite the entire script in a polished, natural, and professional conversational style, clearly labeled by each stage.\n\n"
        "TASK B: For each stage, provide a detailed explanation covering:\n"
        "- Why this stage is crucial and effective in real-world sales interactions.\n"
        "- The key psychological triggers used (e.g., reciprocity, scarcity, authority, social proof), and how they influence the customer.\n"
        "- Which specific phrases or approaches work best at this stage, and why, referencing behavioral science or sales principles.\n\n"
        f"Please answer comprehensively in {lang}, using clear headings for TASK A and TASK B."
    )
    inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = gen_model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=gen_tok.eos_token_id,
        eos_token_id=gen_tok.eos_token_id,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return gen_tok.decode(out[0], skip_special_tokens=True).strip()



class TranscriptResponse(BaseModel):
    greeting: str
    needs: str
    pitch: str
    closing: str
    explanation: str


@app.post("/analyze", response_model=TranscriptResponse)
async def analyze_transcript(file: UploadFile = File(..., description="Provide txt or pdf file")):
    text = (await file.read()).decode("utf-8").strip()
    if not text:
        raise HTTPException(400, "Empty transcript")

    lang = detect(text)

    # Use stage-based chunking for your labeled script
    greeting, needs, pitch, closing = chunk_text_by_stage(text)

    # Summarize each stage
    greeting_sum = summarize_text(greeting, summ_tok, summ_model) if greeting else ""
    needs_sum = summarize_text(needs, summ_tok, summ_model) if needs else ""
    pitch_sum = summarize_text(pitch, summ_tok, summ_model) if pitch else ""
    closing_sum = summarize_text(closing, summ_tok, summ_model) if closing else ""

    explanation = generate_expl(greeting_sum, needs_sum, pitch_sum, closing_sum, lang)

    return TranscriptResponse(
        greeting=greeting_sum,
        needs=needs_sum,
        pitch=pitch_sum,
        closing=closing_sum,
        explanation=explanation,
    )
