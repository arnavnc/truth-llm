import json
import numpy as np
import mlx.core as mx
from mlx_lm_lens.lens import open_lens

MODEL_ID = "mlx-community/Mistral-7B-v0.3-4bit"
model_lens, tokenizer = open_lens(MODEL_ID)

def end_of_block_states(hidden_states):
    n = len(hidden_states)
    if (n - 2) % 2 != 0:
        raise ValueError(f"Unexpected hidden_states length: {n}")
    L = (n - 2) // 2
    return [hidden_states[2*i + 2] for i in range(L)]  # end-of-block states

def sentence_vectors(text: str):
    ids = tokenizer.encode(text)
    out = model_lens(mx.array([ids]), return_dict=True)
    hs = out["hidden_states"]
    blocks = end_of_block_states(hs)

    # Mean pool each layer -> (d,) and cast to float32 for stable numpy ops
    vecs = [mx.mean(b[0], axis=0).astype(mx.float32) for b in blocks]
    mx.eval(*vecs)
    return np.stack([np.array(v) for v in vecs], axis=0)  # (L, d)

def cosine_rows(A, B, eps=1e-12):
    # A,B: (N,d)
    num = np.sum(A * B, axis=1)
    den = (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)) + eps
    return num / den

def main():
    # Load dataset
    rows = []
    with open("data/capitals_pairs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Use TRUE pairs for alignment (cleanest)
    eng_texts = [r["eng_true"] for r in rows]
    fr_texts  = [r["fr_true"] for r in rows]

    # Extract vectors
    E = []
    F = []
    for i, (te, tf) in enumerate(zip(eng_texts, fr_texts)):
        ve = sentence_vectors(te)
        vf = sentence_vectors(tf)
        E.append(ve)
        F.append(vf)
        print(f"Done {i+1}/{len(rows)}", end="\r")

    E = np.stack(E, axis=0)  # (N,L,d)
    F = np.stack(F, axis=0)  # (N,L,d)

    N, L, d = E.shape
    print(f"\nShapes: E={E.shape} F={F.shape}")

    # Raw (uncentered) matched cosine
    raw = np.zeros(L)
    for k in range(L):
        raw[k] = cosine_rows(E[:, k, :], F[:, k, :]).mean()

    # Mean-center per language per layer (removes big shared bias direction)
    muE = E.mean(axis=0)  # (L,d)
    muF = F.mean(axis=0)  # (L,d)
    Ec = E - muE
    Fc = F - muF

    # Centered matched cosine
    matched = np.zeros(L)
    for k in range(L):
        matched[k] = cosine_rows(Ec[:, k, :], Fc[:, k, :]).mean()

    # Shuffled baseline (control): break pairing
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)

    shuffled = np.zeros(L)
    for k in range(L):
        shuffled[k] = cosine_rows(Ec[:, k, :], Fc[perm, k, :]).mean()

    gap = matched - shuffled

    print("\n=== Alignment summary (mean over dataset) ===")
    print("Raw cosine      : min/mean/max =", raw.min(), raw.mean(), raw.max())
    print("Centered matched: min/mean/max =", matched.min(), matched.mean(), matched.max())
    print("Centered shuffled baseline: min/mean/max =", shuffled.min(), shuffled.mean(), shuffled.max())
    print("Gap (matched - shuffled): min/mean/max =", gap.min(), gap.mean(), gap.max())

    best_layer = int(np.argmax(gap))
    print("\nBest layer by gap:", best_layer)
    print("At best layer: matched =", matched[best_layer], "shuffled =", shuffled[best_layer], "gap =", gap[best_layer])

    # Print a few slices so you can see the shape
    print("\nFirst 5 layers gap:", gap[:5])
    mid = L // 2
    print("Middle layers gap:", gap[mid-2:mid+3])
    print("Last 5 layers gap:", gap[-5:])

if __name__ == "__main__":
    main()
