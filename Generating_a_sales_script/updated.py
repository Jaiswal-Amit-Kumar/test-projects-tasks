import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langdetect import detect

# -- Use mLongT5 summarizer and bloomz generator --
SUM_MODEL = "agemagician/mlong-t5-tglobal-base"
GEN_MODEL = "bigscience/bloomz-560m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Sales Script Analyzer", version="4.0")

summ_tok = AutoTokenizer.from_pretrained(SUM_MODEL)
summ_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL).to(device)
gen_tok = AutoTokenizer.from_pretrained(GEN_MODEL)
gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(device)

def summarize_text(text, max_length=150):
    inputs = summ_tok(text, return_tensors="pt", truncation=True, max_length=16384).to(device)
    out = summ_model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return summ_tok.decode(out[0], skip_special_tokens=True)

def chunk_text(text, max_len=1500):
    import re
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) < max_len:
            cur += " " + s
        else:
            chunks.append(cur.strip())
            cur = s
    if cur:
        chunks.append(cur.strip())
    return chunks

def generate_expl(greeting, needs, pitch, closing, lang):
    prompt = (
        "You are an expert multilingual sales coach.\n\n"
        "Here are the four script stages:\n"
        f"1️⃣ Explain Greeting: {greeting}\n"
        f"2️⃣ Explain Needs Discovery: {needs}\n"
        f"3️⃣ Explain Suggestion/Pitch: {pitch}\n"
        f"4️⃣ Explain Closing: {closing}\n\n"
        "**TASK A** – Rewrite the entire dialogue script, stage by stage, in natural, polished conversational style.\n\n"
        "**TASK B** – For each stage, provide:\n"
        "- *Effectiveness:* Why it works in real sales.\n"
        "- *Psychological triggers used* (e.g. reciprocity, scarcity, authority).\n"
        "- *Top phrases* and why they are persuasive (linking back to behavioral principles).\n\n"
        f"Answer in {lang}, with clear 'TASK A' and 'TASK B' sections."
    )
    # ... generation logic continues ...

    inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    out = gen_model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=2,
        eos_token_id=gen_tok.eos_token_id,
        pad_token_id=gen_tok.eos_token_id
    )
    return gen_tok.decode(out[0], skip_special_tokens=True).strip()

class TranscriptResponse(BaseModel):
    greeting: str
    needs: str
    pitch: str
    closing: str
    explanation: str

@app.post("/analyze", response_model=TranscriptResponse)
async def analyze_transcript(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8").strip()
    except:
        raise HTTPException(400, "Please upload a UTF-8 encoded .txt file")
    if not text:
        raise HTTPException(400, "Uploaded file is empty")

    lang = detect(text)
    chunks = chunk_text(text)[:4]
    while len(chunks) < 4:
        chunks.append("")

    greeting = summarize_text(chunks[0])
    needs = summarize_text(chunks[1])
    pitch = summarize_text(chunks[2])
    closing = summarize_text(chunks[3])
    explanation = generate_expl(greeting, needs, pitch, closing, lang)

    return TranscriptResponse(
        greeting=greeting, needs=needs, pitch=pitch,
        closing=closing, explanation=explanation
    )
