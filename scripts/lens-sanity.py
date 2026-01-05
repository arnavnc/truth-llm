import numpy as np
import mlx.core as mx
from mlx_lm_lens.lens import open_lens

MODEL_ID = "mlx-community/Mistral-7B-v0.3-4bit"

# 1) Load model + tokenizer through the "lens" wrapper (so we can inspect internals)
model_lens, tokenizer = open_lens(MODEL_ID)

def forward(text: str):
    token_ids = tokenizer.encode(text)
    tokens = mx.array([token_ids])  # shape: (1, seq_len)
    out = model_lens(tokens, return_dict=True)
    return token_ids, out

def mean_pool_last_layer(hidden_state):
    # hidden_state: shape (1, seq_len, d)
    return mx.mean(hidden_state[0], axis=0)  # shape (d,)

def cosine(u, v):
    return mx.sum(u * v) / (mx.linalg.norm(u) * mx.linalg.norm(v))

def to_f32(x):
    return x.astype(mx.float32)

def stable_norm(x):
    x = to_f32(x)
    return mx.sqrt(mx.sum(x * x))

def stable_cosine(u, v):
    u = to_f32(u)
    v = to_f32(v)
    return mx.sum(u * v) / (stable_norm(u) * stable_norm(v))


eng = "The capital of France is Paris."
fr  = "La capitale de la France est Paris."

# --- Run English once and inspect structure ---
ids_eng, out_eng = forward(eng)

print("\n=== Basic structure check ===")
print("Token count (English):", len(ids_eng))
print("Output keys:", list(out_eng.keys()))

if "hidden_states" not in out_eng:
    raise RuntimeError(
        "No 'hidden_states' returned. "
        "This would mean the lens wrapper isn't returning internals as expected."
    )

hs_eng = out_eng["hidden_states"]
print("Number of hidden_state tensors:", len(hs_eng))
print("First hidden_state shape:", hs_eng[0].shape)
print("Last  hidden_state shape:", hs_eng[-1].shape)

# --- Norm sanity check: last-token vector norm across layers (float32 accumulation) ---
last_token_norms = [stable_norm(h[0, -1, :]) for h in hs_eng]
mx.eval(*last_token_norms)
last_token_norms = np.array([float(n.item()) for n in last_token_norms])

print("\n=== Norm sanity check (English, last token; float32 accumulation) ===")
print("min / mean / max:", last_token_norms.min(), last_token_norms.mean(), last_token_norms.max())

# --- Determinism check: run same input twice, compare final hidden state ---
_, out_eng2 = forward(eng)
hs_eng2 = out_eng2["hidden_states"]

diff = mx.max(mx.abs(hs_eng[-1] - hs_eng2[-1]))
mx.eval(diff)

print("\n=== Determinism check ===")
print("max |Δ| in final-layer hidden states:", float(diff.item()))

# --- Tiny bilingual sanity check: cosine similarity at final layer (mean pooled) ---
ids_fr, out_fr = forward(fr)
hs_fr = out_fr["hidden_states"]

v_eng = mx.mean(hs_eng[-1][0], axis=0)
v_fr  = mx.mean(hs_fr[-1][0], axis=0)

cos = stable_cosine(v_eng, v_fr)
mx.eval(cos)

print("\n=== Tiny bilingual sanity check (float32 cosine) ===")
print("cos(meanpool(final_layer(EN)), meanpool(final_layer(FR))):", float(cos.item()))


cos_self = stable_cosine(v_eng, v_eng)
mx.eval(cos_self)
print("cos(EN, EN) should be ~1.0:", float(cos_self.item()))

