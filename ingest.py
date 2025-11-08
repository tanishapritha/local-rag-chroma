# ingest.py

from sentence_transformers import SentenceTransformer
from db import init_chroma
from ocr import extract_text_from_image

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
collection = init_chroma()


def ingest_text(text: str):
    emb = model.encode([text]).tolist()
    collection.add(
        embeddings=emb,
        documents=[text],
    )
    return {"status": "ok", "ingested_text": text}


def ingest_image(image_bytes: bytes):
    text = extract_text_from_image(image_bytes)
    emb = model.encode([text]).tolist()
    collection.add(
        embeddings=emb,
        documents=[text],
    )
    return {"status": "ok", "extracted_text": text}
