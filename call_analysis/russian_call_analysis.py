import sys
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect
import torch

class CallAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ Using device: {self.device}")
        
        # Load models that don't require authentication
        self.sentiment_pipe = pipeline(
            "text-classification",
            model="blanchefort/rubert-base-cased-sentiment",
            device=self.device
        )
        
        # Alternative translation model that doesn't need sacremoses
        self.translation_pipe = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-ru-en",
            device=self.device
        )
        
        # Using a public model for analysis
        self.analysis_model = AutoModelForSeq2SeqLM.from_pretrained("IlyaGusev/saiga_llama3_8b")
        self.analysis_tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/saiga_llama3_8b")
        self.analysis_model = self.analysis_model.to(self.device)

    def analyze_call(self, transcript_path: str) -> dict:
        """Main analysis function"""
        transcript = self._read_transcript(transcript_path)
        lang = detect(transcript)
        
        print(f"🌐 Detected language: {lang}")
        
        # Ensure we have Russian text
        if lang != "ru":
            translated = self._translate_to_russian(transcript)
        else:
            translated = transcript
            
        sentiment = self._analyze_sentiment(translated)
        analysis = self._generate_analysis(translated)
        
        return {
            "sentiment": sentiment,
            "analysis": analysis,
            "language": lang
        }

    def _read_transcript(self, path: str) -> str:
        """Read transcript from file"""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment with Russian-specific model"""
        print("📊 Analyzing sentiment...")
        result = self.sentiment_pipe(text[:1000])  # Limit to first 1000 chars
        return {
            "label": result[0]["label"],
            "score": result[0]["score"]
        }

    def _generate_analysis(self, text: str) -> str:
        """Generate detailed call analysis in Russian"""
        print("🧠 Generating professional analysis...")
        
        prompt = f"""
Ты - опытный аналитик колл-центра. Проанализируй этот разговор:

{text[:2000]}

Дайте развернутый ответ на русском языке, включая:
1. Оценку работы оператора (1-5)
2. Сильные стороны
3. Рекомендации по улучшению
4. Анализ эмоционального тона
5. Конкретные советы по обучению

Ответ:
"""
        inputs = self.analysis_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        outputs = self.analysis_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        return self.analysis_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _translate_to_russian(self, text: str) -> str:
        """Translate text to Russian if needed"""
        print("🔄 Translating to Russian...")
        return self.translation_pipe(text[:1000])[0]["translation_text"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python call_analyzer.py <transcript_file.txt>")
        return
    
    analyzer = CallAnalyzer()
    results = analyzer.analyze_call(sys.argv[1])
    
    print("\n====================")
    print("📝 Professional Call Analysis")
    print("====================")
    print(f"📊 Sentiment: {results['sentiment']['label']} (confidence: {results['sentiment']['score']:.2f})")
    print(f"🌐 Language: {results['language']}")
    print("\n🔍 Detailed Analysis:")
    print(results["analysis"])
    
    with open("call_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(str(results))

if __name__ == "__main__":
    main()