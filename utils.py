import io, os, tempfile, requests, uuid
from pypdf import PdfReader
from PIL import Image
import pytesseract
from chromadb.api.types import IDs, Metadatas
from config import OLLAMA_GEN_URL, CHAT_MODEL, collection

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
        if end == len(text): break
        start = end - overlap
    return chunks

def llm_answer(context: str, question: str, temperature: float = 0.3) -> str:
    prompt = f"""
You are a helpful assistant.
Return answer in clean Markdown.

Context:
{context}

Question:
{question}

Answer:
""".strip()
    r = requests.post(
        OLLAMA_GEN_URL,
        json={"model": CHAT_MODEL, "prompt": prompt, "options": {"temperature": temperature}, "stream": False},
        timeout=100
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()

def add_chunks(filename: str, text: str, ftype: str):
    chunks = chunk_text(text)
    ids: IDs = [str(uuid.uuid4()) for _ in chunks]
    metas: Metadatas = [{"filename": filename, "type": ftype, "idx": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metas)
    return len(chunks)
