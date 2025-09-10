from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import google.api_core.exceptions
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from typing import List, Dict


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API key securely
genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="AI Agent for Farmers - Animal Husbandry Support")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store session chat history (in-memory)
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

# Request model (single input + optional session_id)
class FarmerQuery(BaseModel):
    query: str
    session_id: str = "default"

# Response model
class FarmerResponse(BaseModel):
    answer: str
    context: str
    note: str = "This is AI-generated information. Please consult a veterinary expert for critical issues."

# Prompt template
HUSBANDRY_PROMPT_TEMPLATE = (
    "You are an expert in animal husbandry. "
    "A farmer has asked the following question:\n\n"
    "{query}\n\n"
    "{previous_context}\n"
    "Give the response in two parts:\n"
    "1. Answer → direct and practical guidance in simple language.\n"
    "2. Context → explain why this answer is important, provide background info, preventive measures, or related best practices.\n"
    "Keep the explanation simple and actionable for farmers."
)

@app.post("/farmer-assistant", response_model=FarmerResponse)
async def farmer_husbandry_assistant(req: FarmerQuery):
    # Retrieve previous context for the session
    history = chat_sessions.get(req.session_id, [])
    previous_context_text = ""
    if history:
        previous_context_text = "Previous conversation:\n" + "\n".join(
            [f"Farmer: {item['query']}\nAI: {item['answer']}" for item in history]
        )

    # Prepare prompt with previous context
    prompt = HUSBANDRY_PROMPT_TEMPLATE.format(
        query=req.query.strip(),
        previous_context=previous_context_text
    )

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        text_output = response.text.strip()

        # Try to split into Answer + Context
        if "Context:" in text_output:
            parts = text_output.split("Context:", 1)
            answer_text = parts[0].replace("Answer:", "").strip()
            context_text = parts[1].strip()
        else:
            answer_text = text_output
            context_text = "This advice is important for improving animal health and productivity in practical farming conditions."

    except google.api_core.exceptions.GoogleAPIError:
        answer_text = "Sorry, I am unable to answer your query right now."
        context_text = "Try again later or consult a nearby veterinary expert."

    # Save this chat in session history
    chat_sessions.setdefault(req.session_id, []).append({
        "query": req.query,
        "answer": answer_text
    })

    return FarmerResponse(answer=answer_text, context=context_text)

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10001)