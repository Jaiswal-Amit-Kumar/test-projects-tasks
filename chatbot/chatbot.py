import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gpt4all import GPT4All

# --- CRM simulation ---
CRM = {
    "cust123": {
        "name": "Anna Smirnova",
        "prev_purchase": "Audi Q7",
        "budget": 200_000,
        "status": "interested"
    },
    "cust456": {
        "name": "Ivan Petrov",
        "prev_purchase": "BMW 3 Series",
        "budget": 80_000,
        "status": "negotiation"
    },
}

# --- Load model correctly ---
model = GPT4All(
    "orca-mini-3b-gguf2-q4_0.gguf",
    allow_download=False,
    n_threads=8,
    device="cpu"
)

app = FastAPI()

class ChatReq(BaseModel):
    customer_id: str
    message: str

@app.post("/chat")
def chat(req: ChatReq):
    cust = CRM.get(req.customer_id)
    if not cust:
        raise HTTPException(status_code=404, detail="Customer not found")

    # Create system prompt to shape personality and behavior
    system_prompt = (
        "### System:\n"
        "You are a professional and friendly car sales assistant. "
        "You always personalize based on CRM data and try to upsell politely.\n\n"
    )

    # Format chat prompt using CRM context
    user_text = (
        f"### User:\n"
        f"Name: {cust['name']}\n"
        f"Previous purchase: {cust['prev_purchase']}\n"
        f"Budget: {cust['budget']}\n"
        f"Status: {cust['status']}\n"
        f"Message: {req.message}\n\n"
        "### Assistant:\n"
    )

    with model.chat_session(system_prompt=system_prompt, prompt_template="{0}"):
        response = model.generate(
            user_text,
            max_tokens=200,
            temp=0.7,
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.18
        )

    return {"reply": response.strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_sales_bot:app", host="0.0.0.0", port=8000)
