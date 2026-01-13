from fastapi import FastAPI
import joblib
from langchain_ollama import ChatOllama
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("sentiment_model.pkl")
llm = ChatOllama(model="gemma3:4b", temperature=0)

class Feedback(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_feedback(item: Feedback):
    # Step 1: Fast ML Prediction
    sentiment = model.predict([item.text])[0]
    
    # Step 2: LLM Explanation (Only if needed to save resources)
    prompt = f"Summarize the main complaint or praise in this feedback in 10 words: {item.text}"
    summary = llm.invoke(prompt).content

    return {
        "sentiment": sentiment,
        "explanation": summary,
        "raw_text": item.text
    }