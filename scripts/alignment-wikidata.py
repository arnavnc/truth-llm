import numpy as np

VEC_PATH = "outputs/wikidata_mistral7b.v0.3_4bit_vecs.npz".replace("v0.3", "v0.3")  # no-op; just to avoid typos
VEC_PATH = "outputs/wikidata_mistral7b_v0.3_4bit_vecs.npz"

def cosine_rows(A, B, eps=1e-12):
    # A,B: (N,d)
    num = np.sum(A * B, axis=1)
    den = (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)) + eps
    return num / den

def alignment_gap(E_pairs, F_pairs, seed=0):
    # E_pairs, F_pairs: (N,L,d) float16/float32
    E = E_pairs.astype(np.float32)
    F = F_pairs.astype(np.float32)
    N, L, d = E.shape

    # mean-center per language per layer
    muE = E.mean(axis=0)  # (L,d)
    muF = F.mean(axis=0)
    Ec = E - muE
    Fc = F - muF

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)

    matched = np.zeros(L, dtype=np.float64)
    shuffled = np.zeros(L, dtype=np.float64)

    for k in range(L):
        matched[k] = cosine_rows(Ec[:, k, :], Fc[:, k, :]).mean()
        shuffled[k] = cosine_rows(Ec[:, k, :], Fc[perm, k, :]).mean()

    gap = matched - shuffled
    return matched, shuffled, gap

def main():
    data = np.load(VEC_PATH, allow_pickle=True)
    E_true  = data["E_true"]
    E_false = data["E_false"]
    F_true  = data["F_true"]
    F_false = data["F_false"]
    relations = data["relations"]

    # Build translation pairs: include TRUE pairs and FALSE pairs (both are valid translations)
    E_pairs = np.concatenate([E_true, E_false], axis=0)  # (2N,L,d)
    F_pairs = np.concatenate([F_true, F_false], axis=0)
    rel_pairs = np.concatenate([relations, relations], axis=0)

    matched, shuffled, gap = alignment_gap(E_pairs, F_pairs, seed=0)
    L = gap.shape[0]

    print("\n=== Overall alignment (centered) ===")
    print("Matched : min/mean/max =", matched.min(), matched.mean(), matched.max())
    print("Shuffled: min/mean/max =", shuffled.min(), shuffled.mean(), shuffled.max())
    print("Gap     : min/mean/max =", gap.min(), gap.mean(), gap.max())
    print("Best layer by gap:", int(np.argmax(gap)))

    # Per-relation breakdown
    print("\n=== Per-relation gap (mean over layers) ===")
    for rel in sorted(set(rel_pairs.tolist())):
        mask = (rel_pairs == rel)
        m2, s2, g2 = alignment_gap(E_pairs[mask], F_pairs[mask], seed=0)
        print(f"{rel:13s} gap mean={g2.mean():.4f}  gap max={g2.max():.4f}  best_layer={int(np.argmax(g2))}")

    # Print slices to see the curve shape
    print("\nFirst 5 gap:", gap[:5])
    mid = L // 2
    print("Middle gap:", gap[mid-2:mid+3])
    print("Last 5 gap:", gap[-5:])

if __name__ == "__main__":
    main()
