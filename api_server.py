import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from main import DialogManager

load_dotenv()

app = FastAPI(title="MLED API Server")

# Allow CORS for Flutter client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    status: str

@app.on_event("startup")
async def startup_event():
    global manager
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("WARNING: DEEPSEEK_API_KEY not set")
    manager = DialogManager(max_tokens=50)
    print("MLED API Server started.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global manager
    if not manager:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if request.message.lower() == 'flush':
        manager._flush_current_chunk()
        return ChatResponse(reply="[System] Flushed current chunk.", status="success")
    elif request.message.lower() == 'forget':
        manager.run_forgetting_cycle()
        return ChatResponse(reply="[System] Ran forgetting cycle.", status="success")
    elif request.message.lower() == 'evolve':
        manager.run_evolution_cycle()
        return ChatResponse(reply="[System] Ran evolution cycle.", status="success")
        
    try:
        reply = manager.process_utterance(request.message)
        if not reply:
            reply = "No response generated."
        return ChatResponse(reply=reply, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
