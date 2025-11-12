from fastapi import FastAPI, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import tempfile, os

from config import chroma_client, collection, CHAT_MODEL
from utils import pdf_to_text, ocr_image, add_chunks, llm_answer

app = FastAPI(title="Pro RAG (PDF+Image)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskBody(BaseModel):
    question: str
    k: int = 4
    temperature: float = 0.1

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    results = []
    for f in files:
        data = await f.read()
        fname = f.filename.lower()
        ctype = (f.content_type or "").lower()
        try:
            if ctype == "application/pdf" or fname.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(data)
                    temp = tmp.name
                try:
                    text = pdf_to_text(temp)
                finally:
                    os.remove(temp)
                ftype = "pdf"
            elif ctype.startswith("image/") or fname.endswith((".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp")):
                text = ocr_image(data)
                ftype = "image"
            else:
                text = data.decode("utf-8", errors="ignore")
                ftype = "text"

            text = (text or "").strip()
            if not text:
                results.append({"filename": f.filename, "status": "empty", "chunks": 0})
                continue

            n = add_chunks(f.filename, text, ftype)
            results.append({"filename": f.filename, "status": "success", "chunks": n})
        except Exception as e:
            results.append({"filename": f.filename, "status": f"error: {e}", "chunks": 0})
    return {"results": results}

@app.get("/stats")
def stats():
    try: count = collection.count()
    except: count = 0
    return {"total_chunks": count, "model": CHAT_MODEL, "embedding": "default MiniLM"}

@app.get("/documents")
def documents(limit: int = 100):
    try: metas = collection.get(include=["metadatas"]).get("metadatas", [])
    except: metas = []
    agg = {}
    for m in metas:
        fn = m.get("filename", "unknown")
        agg[fn] = agg.get(fn, 0) + 1
    items = [{"filename": n, "chunks": agg[n]} for n in sorted(agg.keys())]
    return {"documents": items[:limit]}

@app.get("/search")
def search(q: str = Query(...), k: int = 5):
    res = collection.query(query_texts=[q], n_results=k, include=["documents", "metadatas", "distances"])
    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({
            "snippet": doc[:400] + ("..." if len(doc) > 400 else ""),
            "filename": meta.get("filename", "unknown"),
            "idx": meta.get("idx"),
            "distance": dist
        })
    return {"results": out}

@app.post("/ask")
def ask(body: AskBody):
    res = collection.query(query_texts=[body.question], n_results=body.k, include=["documents", "metadatas"])
    docs, metas = res["documents"][0], res["metadatas"][0]
    if not docs: return {"answer": "I don't know.", "sources": []}
    context = "\n\n".join(docs)
    answer = llm_answer(context, body.question, body.temperature)
    sources = list({m.get("filename", "unknown") for m in metas})
    snippets = [d[:120] + ("..." if len(d) > 120 else "") for d in docs]
    return {"answer": answer, "sources": sources, "snippets": snippets}

@app.post("/reset")
def reset():
    chroma_client.delete_collection("docs")
    global collection
    collection = chroma_client.get_or_create_collection("docs")
    return {"status": "ok"}

app.mount("/app", StaticFiles(directory="static", html=True), name="static")
