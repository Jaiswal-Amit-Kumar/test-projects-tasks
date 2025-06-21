import os
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# ----- Step 1: Transcribe audio if needed -----
def transcribe_audio(audio_path):
    print("🔊 Transcribing audio...")
    whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    result = whisper_pipe(audio_path)
    return result["text"]

# ----- Step 2: Load transcript from file or transcribe -----
def get_transcript(input_path):
    if input_path.endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read()
    elif input_path.endswith(".mp3"):
        return transcribe_audio(input_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .txt or .mp3 file.")

# ----- Step 3: Sentiment analysis -----
def analyze_sentiment(text):
    print("📈 Analyzing sentiment...")
    sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    result = sentiment_model(text[:512])  # Avoid token overflow
    return result[0]

# ----- Step 4: Generate analysis and recommendations -----
def generate_recommendations(text):
    print("🧠 Generating recommendations using TinyLlama...")

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    messages = [
        {"role": "system", "content": "You are an expert call center script analyst."},
        {"role": "user", "content": f"""Analyze the following customer service call transcript.

Transcript:
{text}

Answer the following:
1. Is this a good or bad call? Why?
2. What worked well?
3. What went wrong?
4. Provide 3 clear recommendations to improve the script."""}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("### Response:")[-1].strip() if "### Response:" in result else result

# ----- Main flow -----
def main():
    if len(sys.argv) < 2:
        print("❗ Usage: python analyze_call.py <input_file.mp3|input_file.txt>")
        return

    input_file = sys.argv[1]
    transcript = get_transcript(input_file)
    sentiment = analyze_sentiment(transcript)
    recommendations = generate_recommendations(transcript)

    output_text = (
        "====================\n"
        "✅ Call Analysis Summary\n"
        "====================\n"
        f"📊 Sentiment: {sentiment['label']} (Confidence: {sentiment['score']:.2f})\n\n"
        f"📝 Recommendations and Analysis:\n\n{recommendations}\n"
    )

    print(output_text)

    # Save to file
    with open("call_analysis_output.txt", "w", encoding="utf-8") as f:
        f.write(output_text)

    print("✅ Analysis saved to call_analysis_output.txt")

if __name__ == "__main__":
    main()
