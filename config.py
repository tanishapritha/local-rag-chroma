import chromadb

OLLAMA_GEN_URL = "http://localhost:11434/api/generate"
CHAT_MODEL = "gpt-oss:20b-cloud"

chroma_client = chromadb.PersistentClient(path="chroma")
collection = chroma_client.get_or_create_collection("docs")
