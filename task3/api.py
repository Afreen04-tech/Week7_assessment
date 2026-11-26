"""FastAPI endpoint for the Task 3 policy assistant."""
import os
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import json, numpy as np
BASE=Path(__file__).resolve().parents[1]/'task1'
load_dotenv(BASE/'.env')
app=FastAPI(title='Policy Q&A API')
class AskRequest(BaseModel): question:str
def retrieve(question,k=4):
    store=BASE/'vector_store'
    if not (store/'chunks.json').exists(): raise RuntimeError('Run task1/index.py first')
    chunks=json.loads((store/'chunks.json').read_text(encoding='utf-8')); vectors=np.load(store/'embeddings.npy')
    model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    q=model.encode([question],normalize_embeddings=True)[0]; scores=vectors@q
    ids=np.argsort(scores)[::-1][:k]
    return [(chunks[int(i)],float(scores[int(i)])) for i in ids]
@app.post('/ask')
def ask(body:AskRequest, x_api_key:str=Header(default='')):
    expected=os.getenv('RAG_API_KEY','change-this-key')
    if x_api_key != expected: raise HTTPException(status_code=401,detail='Invalid API key')
    try: results=retrieve(body.question)
    except RuntimeError as e: raise HTTPException(status_code=503,detail=str(e))
    if not results or results[0][1] < 0.28: raise HTTPException(status_code=404,detail="I don't have information on that. No relevant policy chunks found.")
    answer=' '.join(item['text'] for item,score in results[:2])
    sources=[{'source':item['metadata']['source'],'page':item['metadata'].get('page',1)} for item,score in results[:2]]
    return {'answer':answer,'sources':sources}

