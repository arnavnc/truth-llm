import json
import os
import numpy as np
import mlx.core as mx
from tqdm import tqdm
from mlx_lm_lens.lens import open_lens

MODEL_ID = "mlx-community/Mistral-7B-v0.3-4bit"
DATA_PATH = "data/wikidata_enfr_facts_v1.jsonl"
OUT_PATH  = "outputs/wikidata_mistral7b_v0.3_4bit_vecs.npz"

os.makedirs("outputs", exist_ok=True)

model_lens, tokenizer = open_lens(MODEL_ID)

def end_of_block_states(hidden_states):
    n = len(hidden_states)
    if (n - 2) % 2 != 0:
        raise ValueError(f"Unexpected hidden_states length: {n}")
    L = (n - 2) // 2
    return [hidden_states[2*i + 2] for i in range(L)]

def sentence_vectors(text: str) -> np.ndarray:
    ids = tokenizer.encode(text)
    out = model_lens(mx.array([ids]), return_dict=True)
    blocks = end_of_block_states(out["hidden_states"])

    # mean pool each layer to (d,) and store float16 to save disk
    vecs = [mx.mean(b[0], axis=0).astype(mx.float16) for b in blocks]
    mx.eval(*vecs)
    return np.stack([np.array(v) for v in vecs], axis=0)  # (L, d) float16

def main():
    rows = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    N = len(rows)

    # Probe one example to get shapes
    probe = sentence_vectors(rows[0]["eng_true"])
    L, d = probe.shape

    E_true  = np.zeros((N, L, d), dtype=np.float16)
    E_false = np.zeros((N, L, d), dtype=np.float16)
    F_true  = np.zeros((N, L, d), dtype=np.float16)
    F_false = np.zeros((N, L, d), dtype=np.float16)

    relations = []
    s_ids = []

    for i, r in enumerate(tqdm(rows, desc="Extracting vectors")):
        relations.append(r["relation"])
        s_ids.append(r["s_id"])

        E_true[i]  = sentence_vectors(r["eng_true"])
        E_false[i] = sentence_vectors(r["eng_false"])
        F_true[i]  = sentence_vectors(r["fr_true"])
        F_false[i] = sentence_vectors(r["fr_false"])

    np.savez(
        OUT_PATH,
        E_true=E_true, E_false=E_false, F_true=F_true, F_false=F_false,
        relations=np.array(relations),
        s_ids=np.array(s_ids),
        model_id=MODEL_ID
    )
    print(f"Saved vectors to {OUT_PATH}")
    print("Shapes:", E_true.shape, F_true.shape)

if __name__ == "__main__":
    main()
