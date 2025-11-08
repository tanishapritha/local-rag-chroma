from fastapi import FastAPI, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import chromadb
from chromadb.api.types import IDs, Metadatas
import requests
import uuid
from pypdf import PdfReader
import tempfile
import os
from PIL import Image
import pytesseract
import io


# ========= App & CORS =========
app = FastAPI(title="Pro RAG (PDF+Image) with Ollama + Chroma", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========= Chroma (DEFAULT embedding) =========
chroma_client = chromadb.PersistentClient(path="chroma")

# (No embedding_function here — uses Chroma default MiniLM-ONNX)
collection = chroma_client.get_or_create_collection("docs")


# ========= Ollama (LLM) =========
OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
CHAT_MODEL = "gpt-oss:20b-cloud"   # Change if needed


# ========= Utils =========

def pdf_to_text(path: str) -> str:
    text = ""
    pdf = PdfReader(path)
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text


def ocr_image(data: bytes) -> str:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return pytesseract.image_to_string(img)


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 200) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def llm_answer(context: str, question: str, temperature: float = 0.3) -> str:
    prompt = f"""
You are a helpful assistant.
Return answer in clean, well-structured Markdown.

You may use:
- headings
- bullet points
- short paragraphs
- tables

Avoid HTML.
Context:
{context}

Question:
{question}

Answer:
""".strip()


    r = requests.post(
        OLLAMA_GEN_URL,
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "options": {"temperature": temperature},
            "stream": False
        },
        timeout=100
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()


# ========= Schemas =========
class AskBody(BaseModel):
    question: str
    k: int = 4
    temperature: float = 0.1


# ========= Endpoints =========

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    results = []
    for f in files:
        data = await f.read()
        fname = f.filename.lower()
        ctype = (f.content_type or "").lower()

        try:
            # Extract text
            if ctype == "application/pdf" or fname.endswith(".pdf"):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(data)
                    temp = tmp.name
                try:
                    text = pdf_to_text(temp)
                finally:
                    os.remove(temp)

            elif ctype.startswith("image/") or fname.endswith((".jpg", ".jpeg", ".png",".webp",".tiff",".bmp")):
                text = ocr_image(data)

            else:
                text = data.decode("utf-8", errors="ignore")

            text = (text or "").strip()
            if not text:
                results.append({"filename": f.filename, "status": "empty", "chunks": 0})
                continue

            chunks = chunk_text(text)
            ids: IDs = [str(uuid.uuid4()) for _ in chunks]
            metas: Metadatas = [
                {
                    "filename": f.filename,
                    "type": "pdf" if fname.endswith(".pdf") else "image",
                    "idx": i
                }
                for i in range(len(chunks))
            ]

            collection.add(ids=ids, documents=chunks, metadatas=metas)

            results.append({
                "filename": f.filename,
                "status": "success",
                "chunks": len(chunks)
            })

        except Exception as e:
            results.append({"filename": f.filename, "status": f"error: {e}", "chunks": 0})

    return {"results": results}


@app.get("/stats")
def stats():
    try:
        count = collection.count()
    except:
        count = 0
    return {
        "total_chunks": count,
        "model": CHAT_MODEL,
        "embedding": "default (MiniLM ONNX)"
    }


@app.get("/documents")
def documents(limit: int = 100):
    try:
        res = collection.get(include=["metadatas"])
        metas = res.get("metadatas", [])
    except:
        metas = []

    agg: Dict[str, int] = {}
    for m in metas:
        fn = m.get("filename", "unknown")
        agg[fn] = agg.get(fn, 0) + 1

    items = [{"filename": name, "chunks": agg[name]} for name in sorted(agg.keys())]
    return {"documents": items[:limit]}


@app.get("/search")
def search(q: str = Query(...), k: int = 5):
    res = collection.query(
        query_texts=[q],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

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
    res = collection.query(
        query_texts=[body.question],
        n_results=body.k,
        include=["documents", "metadatas"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    if not docs:
        return {"answer": "I don't know.", "sources": []}

    context = "\n\n".join(docs)
    answer = llm_answer(context=context, question=body.question, temperature=body.temperature)

    sources = list({m.get("filename", "unknown") for m in metas})
    snippets = [d[:120] + ("..." if len(d) > 120 else "") for d in docs]

    return {"answer": answer, "sources": sources, "snippets": snippets}


@app.post("/reset")
def reset():
    chroma_client.delete_collection("docs")
    global collection
    collection = chroma_client.get_or_create_collection("docs")
    return {"status": "ok"}


# Serve UI
app.mount("/app", StaticFiles(directory="static", html=True), name="static")
