import uuid
from config import collection, embedding_model

def ingest_text(text: str, filename: str = "manual_text"):
    """Ingest plain text into the collection."""
    if not text.strip():
        return {"status": "error", "message": "Empty text"}
    
    emb = embedding_model.encode([text]).tolist()
    doc_id = str(uuid.uuid4())
    
    # Add to collection
    collection.add(
        ids=[doc_id],
        embeddings=emb,
        documents=[text],
        metadatas=[{"filename": filename, "type": "text", "idx": 0}]
    )
    
    return {"status": "ok", "ingested_text": text, "id": doc_id}


def ingest_image(image_bytes: bytes, filename: str = "manual_image"):
    """Extract text from image and ingest."""
    from PIL import Image
    import pytesseract
    import io
    
    if not image_bytes:
        return {"status": "error", "message": "Empty image"}
    
    # OCR
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = pytesseract.image_to_string(img).strip()
    
    if not text:
        return {"status": "error", "message": "No text extracted"}
    
    emb = embedding_model.encode([text]).tolist()

    doc_id = str(uuid.uuid4())
    
    collection.add(
        ids=[doc_id],
        embeddings=emb,
        documents=[text],
        metadatas=[{"filename": filename, "type": "image", "idx": 0}]
    )
    
    return {"status": "ok", "extracted_text": text, "id": doc_id}