import argparse
import json
import os
import numpy as np

VEC_PATH_DEFAULT = "outputs/wikidata_mistral7b_v0.3_4bit_vecs.npz"


def cosine_rows(A, B, eps=1e-12):
    num = np.sum(A * B, axis=1)
    den = (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)) + eps
    return num / den

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v) + eps
    return v / n

def end_to_end_pca_basis(X, r):
    # economy SVD; since n_samples << d, Vt has shape (n_samples, d)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    B = Vt[:r].T  # (d, r)
    return B

def procrustes_row_mapping(E, F, proper=True):
    # E,F: (n,r) row-form; find W (r,r) s.t. E W ≈ F
    M = E.T @ F
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    W = U @ Vt
    if not proper:
        return W
    # enforce det(W)=+1 (proper rotation)
    if np.linalg.det(W) < 0:
        S = np.eye(U.shape[1])
        S[-1, -1] = -1
        W = U @ S @ Vt
    return W

def split_indices_random(idx_pool, test_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.array(idx_pool, dtype=int)
    rng.shuffle(idx)
    n_test = int(np.round(len(idx) * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx

def split_indices_stratified(relations, idx_pool, test_frac=0.2, seed=0):
    """
    Stratify by relation label so each relation contributes ~test_frac to test set.
    relations: array-like of length N
    idx_pool: indices into relations defining the pool
    """
    rng = np.random.default_rng(seed)
    idx_pool = np.array(idx_pool, dtype=int)
    rel_pool = np.array(relations)[idx_pool]

    train_idx = []
    test_idx = []

    for rel in sorted(set(rel_pool.tolist())):
        rel_idxs = idx_pool[rel_pool == rel]
        rel_idxs = rel_idxs.copy()
        rng.shuffle(rel_idxs)
        n_test = int(np.round(len(rel_idxs) * test_frac))
        test_idx.extend(rel_idxs[:n_test].tolist())
        train_idx.extend(rel_idxs[n_test:].tolist())

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)

def pairwise_accuracy(X_true, X_false, w):
    s_true = X_true @ w
    s_false = X_false @ w
    return float(np.mean(s_true > s_false))

def mean_margin(X_true, X_false, w):
    return float(np.mean((X_true - X_false) @ w))

def parse_rel_list(s: str):
    s = s.strip()
    if s.lower() in ["all", "*", ""]:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]

def append_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec_path", type=str, default=VEC_PATH_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--r", type=int, default=256)
    ap.add_argument("--layers", type=str, default="all",
                    help="Comma-separated layer indices (e.g., 0,22,24) or 'all'")

    # Output + experiment toggles
    ap.add_argument("--out_jsonl", type=str, default=None,
                    help="If set, append one JSON row per layer (and optionally per relation).")
    ap.add_argument("--tag", type=str, default="base",
                    help="Condition label written into output rows (e.g., base, shuffled).")
    ap.add_argument("--shuffle_train_pairs", action="store_true",
                    help="CONTROL: shuffle EN↔FR pairing when fitting W (should kill the effect).")
    ap.add_argument("--emit_by_relation", action="store_true",
                    help="If set, emit extra rows per relation (plus __all__).")
    ap.add_argument("--stratify_relations", action="store_true",
                    help="If set, do stratified train/test split by relation.")

    # Optional generalization test
    ap.add_argument("--train_relations", type=str, default="all",
                    help="Comma-separated relation names to use for training W and w_eng (default: all).")
    ap.add_argument("--test_relations", type=str, default="all",
                    help="Comma-separated relation names to use for evaluation (default: all). "
                         "If not 'all', evaluation set is fixed to these relations (no random split).")

    args = ap.parse_args()

    data = np.load(args.vec_path, allow_pickle=True)
    E_true  = data["E_true"]    # (N,L,d) float16
    E_false = data["E_false"]
    F_true  = data["F_true"]
    F_false = data["F_false"]
    relations = np.array(data["relations"]).astype(str)  # (N,)

    N, L, d = E_true.shape

    # Layer selection
    if args.layers.strip().lower() == "all":
        layers = list(range(L))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    # Relation filtering
    train_rels = parse_rel_list(args.train_relations)
    test_rels  = parse_rel_list(args.test_relations)

    idx_all = np.arange(N, dtype=int)

    train_pool = idx_all if train_rels is None else idx_all[np.isin(relations, train_rels)]
    if test_rels is None:
        # normal split mode
        if args.stratify_relations:
            train_idx, test_idx = split_indices_stratified(relations, train_pool, args.test_frac, args.seed)
        else:
            train_idx, test_idx = split_indices_random(train_pool, args.test_frac, args.seed)
    else:
        # holdout/generalization mode: fixed eval set
        test_idx = idx_all[np.isin(relations, test_rels)]
        # training excludes eval indices by default
        train_idx = np.setdiff1d(train_pool, test_idx, assume_unique=False)

    # PCA rank cap based on training sample count
    n_train_pairs = 2 * len(train_idx)  # true+false rows per language
    max_r = min(args.r, (2 * n_train_pairs) - 1, d)  # PCA on concat(E,F) has 2*n_train_pairs rows
    r = int(max_r)

    print(f"N={N}, L={L}, d={d}. Train rows={len(train_idx)}, Test rows={len(test_idx)}. Using r={r}")
    print(f"Condition tag={args.tag} | shuffle_train_pairs={args.shuffle_train_pairs} | stratify_relations={args.stratify_relations}")
    if train_rels is not None:
        print("Train relations:", train_rels)
    if test_rels is not None:
        print("Test relations:", test_rels)

    out_rows = []

    for k in layers:
        # --- Build train matrices (translation pairs) ---
        Etr = np.concatenate([E_true[train_idx, k, :], E_false[train_idx, k, :]], axis=0).astype(np.float32)
        Ftr = np.concatenate([F_true[train_idx, k, :], F_false[train_idx, k, :]], axis=0).astype(np.float32)

        # Center per language (train only)
        muE = Etr.mean(axis=0)
        muF = Ftr.mean(axis=0)
        Etrc = Etr - muE
        Ftrc = Ftr - muF

        # PCA basis on combined centered train data
        X_for_pca = np.concatenate([Etrc, Ftrc], axis=0)
        B = end_to_end_pca_basis(X_for_pca, r)  # (d,r)

        # Project train
        Etr_r = Etrc @ B
        Ftr_r = Ftrc @ B

        # CONTROL: shuffle pairing for Procrustes fit
        if args.shuffle_train_pairs:
            rng = np.random.default_rng(args.seed + 1000 + k)
            perm = rng.permutation(Ftr_r.shape[0])
            Ftr_r_used = Ftr_r[perm]
        else:
            Ftr_r_used = Ftr_r

        # Procrustes mapping E W ≈ F (row form)
        W = procrustes_row_mapping(Etr_r, Ftr_r_used, proper=True)

        # --- Test translation alignment (held-out) ---
        Ete = np.concatenate([E_true[test_idx, k, :], E_false[test_idx, k, :]], axis=0).astype(np.float32)
        Fte = np.concatenate([F_true[test_idx, k, :], F_false[test_idx, k, :]], axis=0).astype(np.float32)

        Ete_r = (Ete - muE) @ B
        Fte_r = (Fte - muF) @ B

        align_before = float(cosine_rows(Ete_r, Fte_r).mean())
        align_after  = float(cosine_rows(Ete_r @ W, Fte_r).mean())
        align_delta  = align_after - align_before

        # --- Truth directions from train ---
        Etrue_tr  = (E_true[train_idx, k, :].astype(np.float32) - muE) @ B
        Efalse_tr = (E_false[train_idx, k, :].astype(np.float32) - muE) @ B
        delta_eng = Etrue_tr - Efalse_tr
        w_eng = normalize(delta_eng.mean(axis=0))

        Ftrue_tr  = (F_true[train_idx, k, :].astype(np.float32) - muF) @ B
        Ffalse_tr = (F_false[train_idx, k, :].astype(np.float32) - muF) @ B
        delta_fr = Ftrue_tr - Ffalse_tr
        w_fr_star = normalize(delta_fr.mean(axis=0))

        # Predicted French direction via rotation (weights transform is W^T)
        w_fr_hat = normalize(W.T @ w_eng)

        # --- Test vectors (examples, not doubled rows) ---
        Etrue_te  = (E_true[test_idx, k, :].astype(np.float32) - muE) @ B
        Efalse_te = (E_false[test_idx, k, :].astype(np.float32) - muE) @ B
        Ftrue_te  = (F_true[test_idx, k, :].astype(np.float32) - muF) @ B
        Ffalse_te = (F_false[test_idx, k, :].astype(np.float32) - muF) @ B

        # Metrics (overall)
        acc_eng = pairwise_accuracy(Etrue_te, Efalse_te, w_eng)
        acc_fr_unal = pairwise_accuracy(Ftrue_te, Ffalse_te, w_eng)
        acc_fr_al   = pairwise_accuracy(Ftrue_te, Ffalse_te, w_fr_hat)
        acc_fr_star = pairwise_accuracy(Ftrue_te, Ffalse_te, w_fr_star)

        m_fr_unal = mean_margin(Ftrue_te, Ffalse_te, w_eng)
        m_fr_al   = mean_margin(Ftrue_te, Ffalse_te, w_fr_hat)
        m_fr_star = mean_margin(Ftrue_te, Ffalse_te, w_fr_star)

        cos_raw = float((w_eng @ w_fr_star) / (np.linalg.norm(w_eng) * np.linalg.norm(w_fr_star) + 1e-12))
        cos_w   = float((w_fr_hat @ w_fr_star) / (np.linalg.norm(w_fr_hat) * np.linalg.norm(w_fr_star) + 1e-12))

        print(
            f"Layer {k:02d} | align {align_before:.3f}->{align_after:.3f} (Δ {align_delta:+.3f})"
            f" | acc EN {acc_eng:.3f}"
            f" | FR unal {acc_fr_unal:.3f}  FR al {acc_fr_al:.3f}  FR* {acc_fr_star:.3f}"
            f" | cos(ŵ_fr, w*_fr) {cos_w:.3f}"
        )

        # Build output rows
        base_row = {
            "seed": args.seed,
            "tag": args.tag,
            "layer": int(k),
            "relation": "__all__",

            "align_before": align_before,
            "align_after": align_after,
            "align_delta": align_delta,

            "acc_eng": acc_eng,
            "acc_fr_unaligned": acc_fr_unal,
            "acc_fr_aligned": acc_fr_al,
            "acc_fr_upper": acc_fr_star,

            "m_fr_unal": m_fr_unal,
            "m_fr_al": m_fr_al,
            "m_fr_star": m_fr_star,

            "cos_raw_weng_wfrstar": cos_raw,
            "cos_wfrhat_wfrstar": cos_w,

            "r": r,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "shuffle_train_pairs": bool(args.shuffle_train_pairs),
        }
        out_rows.append(base_row)

        if args.emit_by_relation:
            # per relation metrics on test set
            rel_test = relations[test_idx]
            for rel in sorted(set(rel_test.tolist())):
                mask = (rel_test == rel)
                if mask.sum() == 0:
                    continue

                # relation-specific alignment: rebuild doubled rows for this subset
                idx_rel = test_idx[mask]
                Ete_rel = np.concatenate([E_true[idx_rel, k, :], E_false[idx_rel, k, :]], axis=0).astype(np.float32)
                Fte_rel = np.concatenate([F_true[idx_rel, k, :], F_false[idx_rel, k, :]], axis=0).astype(np.float32)
                Ete_rel_r = (Ete_rel - muE) @ B
                Fte_rel_r = (Fte_rel - muF) @ B
                ab = float(cosine_rows(Ete_rel_r, Fte_rel_r).mean())
                aa = float(cosine_rows(Ete_rel_r @ W, Fte_rel_r).mean())
                ad = aa - ab

                # subset truth metrics
                acc_fr_unal_rel = pairwise_accuracy(Ftrue_te[mask], Ffalse_te[mask], w_eng)
                acc_fr_al_rel   = pairwise_accuracy(Ftrue_te[mask], Ffalse_te[mask], w_fr_hat)
                acc_fr_star_rel = pairwise_accuracy(Ftrue_te[mask], Ffalse_te[mask], w_fr_star)

                m_fr_unal_rel = mean_margin(Ftrue_te[mask], Ffalse_te[mask], w_eng)
                m_fr_al_rel   = mean_margin(Ftrue_te[mask], Ffalse_te[mask], w_fr_hat)
                m_fr_star_rel = mean_margin(Ftrue_te[mask], Ffalse_te[mask], w_fr_star)

                rel_row = dict(base_row)
                rel_row.update({
                    "relation": rel,
                    "n_test": int(mask.sum()),
                    "align_before": ab,
                    "align_after": aa,
                    "align_delta": ad,
                    "acc_fr_unaligned": acc_fr_unal_rel,
                    "acc_fr_aligned": acc_fr_al_rel,
                    "acc_fr_upper": acc_fr_star_rel,
                    "m_fr_unal": m_fr_unal_rel,
                    "m_fr_al": m_fr_al_rel,
                    "m_fr_star": m_fr_star_rel,
                })
                out_rows.append(rel_row)

    # Write JSONL
    if args.out_jsonl is not None:
        append_jsonl(args.out_jsonl, out_rows)
        print("Appended rows to:", args.out_jsonl)


if __name__ == "__main__":
    main()
