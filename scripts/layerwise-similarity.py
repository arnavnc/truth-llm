import numpy as np
import mlx.core as mx
from mlx_lm_lens.lens import open_lens

MODEL_ID = "mlx-community/Mistral-7B-v0.3-4bit"
model_lens, tokenizer = open_lens(MODEL_ID)

def to_f32(x):
    return x.astype(mx.float32)

def stable_norm(x):
    x = to_f32(x)
    return mx.sqrt(mx.sum(x * x))

def stable_cosine(u, v):
    u = to_f32(u)
    v = to_f32(v)
    return mx.sum(u * v) / (stable_norm(u) * stable_norm(v))

def forward_hidden_states(text: str):
    token_ids = tokenizer.encode(text)
    tokens = mx.array([token_ids])  # (1, T)
    out = model_lens(tokens, return_dict=True)
    return token_ids, out["hidden_states"]

def mean_pool(H):
    # H: (1, T, d) -> (d,)
    return mx.mean(H[0], axis=0)

def end_of_block_states(hidden_states):
    """
    Convert the lens' hidden_states list into one tensor per transformer block (layer).
    Expected pattern when len = 2*L + 2:
      0 = embeddings
      2i+1 = after attention in layer i
      2i+2 = after MLP / end of block in layer i
      last = final norm/output
    We return the end-of-block states for each layer.
    """
    n = len(hidden_states)
    # infer L from n = 2L + 2
    if (n - 2) % 2 != 0:
        raise ValueError(f"Unexpected hidden_states length: {n}")
    L = (n - 2) // 2

    # indices 2,4,6,...,2L
    block_states = [hidden_states[2*i + 2] for i in range(L)]
    return block_states  # length L

eng = "The capital of France is Paris."
fr  = "La capitale de la France est Paris."

ids_eng, hs_eng = forward_hidden_states(eng)
ids_fr,  hs_fr  = forward_hidden_states(fr)

blocks_eng = end_of_block_states(hs_eng)
blocks_fr  = end_of_block_states(hs_fr)

print(f"Token counts: EN={len(ids_eng)} FR={len(ids_fr)}")
print(f"Hidden states total: {len(hs_eng)}")
print(f"Blocks inferred: {len(blocks_eng)}")
print(f"Block state shape: {blocks_eng[0].shape} (should be (1, T, d))")

# Layerwise cosine similarity
cos_by_layer = []
for k in range(len(blocks_eng)):
    v_eng = mean_pool(blocks_eng[k])
    v_fr  = mean_pool(blocks_fr[k])
    c = stable_cosine(v_eng, v_fr)
    mx.eval(c)
    cos_by_layer.append(float(c.item()))

cos_by_layer = np.array(cos_by_layer)

print("\n=== Layerwise EN–FR cosine similarity (end-of-block, mean pooled) ===")
print("First 5:", cos_by_layer[:5])
print("Middle 5:", cos_by_layer[len(cos_by_layer)//2 - 2 : len(cos_by_layer)//2 + 3])
print("Last 5:", cos_by_layer[-5:])
print("\nSummary: min / mean / max =", cos_by_layer.min(), cos_by_layer.mean(), cos_by_layer.max())
print("Best layer index:", int(np.argmax(cos_by_layer)))
