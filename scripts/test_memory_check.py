# test_memory_check.py
# 放在 E:\Projects\rag_project\ 下執行

import chromadb
import config as cfg

client = chromadb.PersistentClient(path=cfg.MEMORY_DB_DIR)

episodic = client.get_collection(cfg.MEMORY_COLLECTION_EPISODIC)
preference = client.get_collection(cfg.MEMORY_COLLECTION_PREFERENCE)

print(f"\n=== episodic_memory：{episodic.count()} 筆 ===")
if episodic.count() > 0:
    results = episodic.get()
    for doc, meta in zip(results["documents"], results["metadatas"]):
        print(f"\n  type: {meta.get('type')}")
        print(f"  date: {meta.get('date')}")
        print(f"  內容: {doc[:150]}...")

print(f"\n=== preference_memory：{preference.count()} 筆 ===")
if preference.count() > 0:
    results = preference.get()
    for doc, meta in zip(results["documents"], results["metadatas"]):
        print(f"\n  type: {meta.get('type')}")
        print(f"  date: {meta.get('date')}")
        print(f"  內容: {doc[:150]}...")
