import torch, json

E = torch.load("data/processed/index/embeddings.pt")
meta = [json.loads(l) for l in open("data/processed/index/meta.jsonl", encoding="utf-8")]

sims = E[:50] @ E.T
best = sims.argmax(dim=1)
bad = [(i, best[i].item(), sims[i, best[i]].item(), sims[i, i].item())
       for i in range(50) if best[i].item() != i]

print(f"{len(bad)} of 50 mismatched\n")
for i, j, s_best, s_self in bad[:5]:
    print(f"chunk {i} -> {j}   best={s_best:.6f}  self={s_self:.6f}")
    print(f"   {i}: {meta[i]['chunk_id']}")
    print(f"   {j}: {meta[j]['chunk_id']}\n")

chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
texts = [c["text"] for c in chunks]
print("unique texts:", len(set(texts)), "of", len(texts))