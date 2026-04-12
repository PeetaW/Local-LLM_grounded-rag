import requests

payload = {
    "model": "bge-m3",
    "prompt": "What is the synthesis method of NZVI?"
}

r = requests.post("http://localhost:11434/api/embeddings", json=payload)
print(f"HTTP 狀態碼: {r.status_code}")
data = r.json()
embedding = data.get("embedding", [])
print(f"embedding 長度: {len(embedding)}")
print(f"前5個值: {embedding[:5]}")