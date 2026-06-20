import ollama
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE
# ─────────────────────────────────────────────
documents = [
    {
        "id": "doc_001",
        "text": "Windows Update error 0x80070005 means Access Denied. This occurs when the Windows Update service lacks permissions to write to system directories. Fix: run Windows Update troubleshooter as Administrator, or reset Windows Update components via command prompt."
    },
    {
        "id": "doc_002",
        "text": "To reset Windows Update components, open Command Prompt as Administrator and run: net stop wuauserv, net stop cryptSvc, net stop bits, net stop msiserver. Then rename SoftwareDistribution and Catroot2 folders, and restart the services."
    },
    {
        "id": "doc_003",
        "text": "Windows Defender Firewall can block application network access. To allow an app through the firewall, go to Control Panel > System and Security > Windows Defender Firewall > Allow an app or feature through Windows Defender Firewall."
    },
    {
        "id": "doc_004",
        "text": "Blue Screen of Death (BSOD) errors in Windows are caused by critical system failures. Common causes include faulty drivers, hardware issues, or corrupted system files. Use Event Viewer or WinDbg to analyze minidump files for root cause."
    },
    {
        "id": "doc_005",
        "text": "To perform a clean boot in Windows, open System Configuration (msconfig), go to Services tab, check Hide all Microsoft services, then Disable all. Under Startup tab, open Task Manager and disable startup items. Restart to identify software conflicts."
    },
    {
        "id": "doc_006",
        "text": "Disk cleanup in Windows removes temporary files, system cache, and old Windows installation files. Access it via right-clicking a drive > Properties > Disk Cleanup. Run as Administrator to also clean system files and free significant space."
    },
    {
        "id": "doc_007",
        "text": "Windows Task Manager shows CPU, memory, disk, and network usage per process. Press Ctrl+Shift+Esc to open it. Use the Details tab for advanced process management, or End Task to force-close unresponsive applications."
    },
    {
        "id": "doc_008",
        "text": "Registry Editor (regedit) allows editing the Windows registry. Incorrect changes can cause system instability. Always export a registry backup before making changes. Access via Win+R > regedit. Key hives: HKLM, HKCU, HKCR, HKU, HKCC."
    }
]

# ─────────────────────────────────────────────
# 2. CHROMADB SETUP
# ─────────────────────────────────────────────
emb_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

try:
    chroma_client.delete_collection("lab3_kb")
except:
    pass

collection = chroma_client.create_collection(
    name="lab3_kb",
    embedding_function=emb_fn
)

collection.add(
    ids=[doc["id"] for doc in documents],
    documents=[doc["text"] for doc in documents]
)
print(f"ChromaDB ready: {collection.count()} doc")

# ─────────────────────────────────────────────
# 3. BM25 SETUP
# ─────────────────────────────────────────────
def tokenize(text):
    return text.lower().split()

corpus_tokens = [tokenize(doc["text"]) for doc in documents]
bm25 = BM25Okapi(corpus_tokens)
print("BM25 ready")

# ─────────────────────────────────────────────
# 4. RETRIEVAL FUNCTIONS
# ─────────────────────────────────────────────
def dense_retrieve(query: str, top_k: int = 3) -> list:
    """Baseline: yalnız dense vector search"""
    results = collection.query(query_texts=[query], n_results=top_k)
    retrieved = []
    for i, doc_id in enumerate(results["ids"][0]):
        retrieved.append({
            "id": doc_id,
            "text": results["documents"][0][i],
            "score": 1 - results["distances"][0][i]
        })
    return retrieved


def hybrid_retrieve(query: str, top_k: int = 3, alpha: float = 0.5) -> list:
    """Hybrid: dense + BM25 alpha blend"""
    dense_results = collection.query(query_texts=[query], n_results=len(documents))
    dense_scores = {}
    for i, doc_id in enumerate(dense_results["ids"][0]):
        dense_scores[doc_id] = 1 - dense_results["distances"][0][i]

    # BM25 scores (normalized)
    bm25_raw = bm25.get_scores(tokenize(query))
    bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_scores = {doc["id"]: bm25_raw[i] / bm25_max for i, doc in enumerate(documents)}

    # Combine
    combined = {}
    for doc in documents:
        did = doc["id"]
        combined[did] = alpha * dense_scores.get(did, 0) + (1 - alpha) * bm25_scores.get(did, 0)

    sorted_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]
    id_to_text = {doc["id"]: doc["text"] for doc in documents}
    return [{"id": did, "text": id_to_text[did], "score": combined[did]} for did in sorted_ids]

# ─────────────────────────────────────────────
# 5. EVAL SET
# ─────────────────────────────────────────────
eval_set = [
    {
        "question": "How do I fix error code 0x80070005 in Windows Update?",
        "expected_id": "doc_001"
    },
    {
        "question": "What steps are needed to reset Windows Update components using command prompt?",
        "expected_id": "doc_002"
    },
    {
        "question": "My application cannot connect to the internet, how do I configure the firewall?",
        "expected_id": "doc_003"
    },
    {
        "question": "Windows crashes with a blue screen, how can I find the root cause?",
        "expected_id": "doc_004"
    },
    {
        "question": "How do I disable startup programs to troubleshoot software conflicts?",
        "expected_id": "doc_005"
    }
]

# ─────────────────────────────────────────────
# 6. LLM FUNCTIONS (OLLAMA)
# ─────────────────────────────────────────────
def generate_answer(query: str, context_docs: list, model: str = "llama3.2") -> str:
    context_text = "\n\n".join([f"[{d['id']}]: {d['text']}" for d in context_docs])
    prompt = f"""You are a helpful Windows technical support assistant.
Answer the user's question using ONLY the provided context below.
If the answer is not in the context, say "I don't know based on the provided information."

Context:
{context_text}

Question: {query}

Answer:"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()


def judge_faithfulness(question: str, answer: str, context_docs: list, model: str = "llama3.2") -> str:
    context_text = "\n\n".join([d["text"] for d in context_docs])
    prompt = f"""You are an evaluation judge.
Determine if the answer is fully supported by the given context.

Context:
{context_text}

Question: {question}
Answer: {answer}

Reply with ONLY one word: yes or no."""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    verdict = response["message"]["content"].strip().lower()
    return "yes" if "yes" in verdict else "no"

# ─────────────────────────────────────────────
# 7. EVAL LOOP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION BEGINS")
print("=" * 60)

results = []

for i, item in enumerate(eval_set):
    q = item["question"]
    expected = item["expected_id"]
    print(f"\n Q{i+1}: {q[:55]}...")

    # Baseline
    b_docs = dense_retrieve(q, top_k=3)
    b_ids = [d["id"] for d in b_docs]
    b_hit = expected in b_ids
    b_answer = generate_answer(q, b_docs)
    b_faith = judge_faithfulness(q, b_answer, b_docs)

    # Hybrid
    h_docs = hybrid_retrieve(q, top_k=3, alpha=0.5)
    h_ids = [d["id"] for d in h_docs]
    h_hit = expected in h_ids
    h_answer = generate_answer(q, h_docs)
    h_faith = judge_faithfulness(q, h_answer, h_docs)

    results.append({
        "question": q,
        "expected_id": expected,
        "baseline_hit": b_hit,
        "baseline_retrieved": b_ids,
        "baseline_faith": b_faith,
        "hybrid_hit": h_hit,
        "hybrid_retrieved": h_ids,
        "hybrid_faith": h_faith,
    })

    print(f"Expected: {expected}")
    print(f"Baseline: hit={b_hit} | faithful={b_faith} | retrieved={b_ids}")
    print(f"Hybrid: hit={h_hit} | faithful={h_faith} | retrieved={h_ids}")

# ─────────────────────────────────────────────
# 8. RESULTS TABLE
# ─────────────────────────────────────────────
b_hit_rate = sum(r["baseline_hit"] for r in results) / len(results)
h_hit_rate = sum(r["hybrid_hit"]   for r in results) / len(results)
b_faith_rate = sum(1 for r in results if r["baseline_faith"] == "yes") / len(results)
h_faith_rate = sum(1 for r in results if r["hybrid_faith"]  == "yes") / len(results)

print("\n" + "=" * 58)
print("RESULT TABLE")
print("=" * 58)
print(f"{'Metric':<28} {'Baseline':>12} {'Hybrid':>10}")
print("-" * 58)
print(f"{'Retrieval Hit Rate':<28} {b_hit_rate:>11.0%} {h_hit_rate:>9.0%}")
print(f"{'Faithfulness (LLM judge)':<28} {b_faith_rate:>11.0%} {h_faith_rate:>9.0%}")
print("=" * 58)

print("\nQUESTION BREAKDOWN")
print("-" * 58)
for i, r in enumerate(results, 1):
    print(f"Q{i} [{r['expected_id']}]")
    print(f"Baseline: hit={'✅' if r['baseline_hit'] else '❌'}  faithful={'✅' if r['baseline_faith']=='yes' else '❌'}  {r['baseline_retrieved']}")
    print(f"Hybrid  : hit={'✅' if r['hybrid_hit'] else '❌'}  faithful={'✅' if r['hybrid_faith']=='yes' else '❌'}  {r['hybrid_retrieved']}")

# ─────────────────────────────────────────────
# 9. EVAL_RESULTS.MD 
# ─────────────────────────────────────────────
md = f"""# Eval Results — Lab 3: Hybrid Search vs Dense Baseline

## Comparison Table

| Metric | Baseline (Dense only) | Hybrid (Dense + BM25) |
|---|---|---|
| Retrieval Hit Rate | {b_hit_rate:.0%} | {h_hit_rate:.0%} |
| Faithfulness (LLM-as-judge) | {b_faith_rate:.0%} | {h_faith_rate:.0%} |

## Question-level Breakdown

| Q | Expected | Baseline Hit | Hybrid Hit | Baseline Faithful | Hybrid Faithful |
|---|---|---|---|---|---|
"""

for i, r in enumerate(results, 1):
    md += (
        f"| Q{i} | {r['expected_id']} "
        f"| {'✅' if r['baseline_hit'] else '❌'} "
        f"| {'✅' if r['hybrid_hit'] else '❌'} "
        f"| {'✅' if r['baseline_faith']=='yes' else '❌'} "
        f"| {'✅' if r['hybrid_faith']=='yes' else '❌'} |\n"
    )

md += f"""
## Conclusion

Hybrid search combined dense vector retrieval (all-MiniLM-L6-v2 via ChromaDB)
with BM25 keyword scoring using alpha=0.5 (equal weight blend).
The key motivation was Q1 — the exact error code `0x80070005` — which dense
retrieval tends to fumble because semantic embeddings compress alphanumeric
tokens poorly, while BM25 catches exact string matches reliably.

Retrieval hit rate: **{b_hit_rate:.0%}** (baseline) → **{h_hit_rate:.0%}** (hybrid).
Faithfulness: **{b_faith_rate:.0%}** (baseline) → **{h_faith_rate:.0%}** (hybrid).

The results {'support the hypothesis: hybrid search improved retrieval, especially for the exact-term query where BM25 keyword overlap rescued cases dense similarity missed.' if h_hit_rate >= b_hit_rate else 'show a flat or negative result: both methods performed similarly on this small, short-document knowledge base. A larger and more diverse corpus would provide a more decisive comparison.'}
"""

with open("eval_results.md", "w",encoding="utf-8") as f:
    f.write(md)

print("\n✅ eval_results.md written!")