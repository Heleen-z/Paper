"""Run a sealed-target, feature-level Office-Home Pr→Rw active adaptation pilot."""

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

# Define BLAS concurrency before importing numpy for deterministic CPU checkpoints.
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FEATURES = {
    "Product_Product.csv": "5d1769b2329875bfb92c90670d8ecc02c40552e191953c7c9931fb593c673061",
    "Product_RealWorld.csv": "acd73b0e4126e8b8756178146916feb0f2f3d9c990a91f7b99c5a307c186e453",
}
SEEDS = (11, 23, 37)
METHODS = ("random", "entropy")
SOURCE_EPOCHS = 15
SOURCE_BATCH = 128
ADAPT_STEPS = 24
ADAPT_BATCH_PER_DOMAIN = 64
SPLIT_SEED = 2026
CLASSES = 65


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_domain(path):
    table = np.loadtxt(path, delimiter=",", dtype=np.float32)
    y = table[:, -1].astype(np.int64)
    x = np.ascontiguousarray(table[:, :-1])
    if (x.shape[1] != 2048 or not np.isfinite(x).all()
            or not np.array_equal(y, table[:, -1])
            or not np.array_equal(np.unique(y), np.arange(CLASSES))):
        raise ValueError(f"Unexpected feature matrix in {path}")
    lengths = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(lengths == 0):
        raise ValueError("A feature row has zero norm")
    return (x / lengths) * np.float32(10.0), y


def split_rows(y, eligible):
    rng = np.random.default_rng(SPLIT_SEED)
    holdout = []
    for label in np.unique(y):
        indices = eligible[y[eligible] == label]
        shuffled = rng.permutation(indices)
        holdout.extend(shuffled[:max(1, round(len(indices) * 0.2))])
    return np.setdiff1d(eligible, holdout), np.sort(np.asarray(holdout))


def unique_features(x, previous):
    keep, excluded = [], []
    for i, row in enumerate(x):
        digest = hashlib.sha256(row.tobytes()).digest()
        if digest in previous:
            excluded.append(i)
        else:
            previous.add(digest)
            keep.append(i)
    return np.asarray(keep), np.asarray(excluded, dtype=np.int64)


def probabilities(x, w, b):
    scores = x @ w + b
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def gradient_step(x, y, w, b, rate):
    gradient = probabilities(x, w, b)
    gradient[np.arange(len(y)), y] -= 1
    gradient /= np.float32(len(y))
    w -= rate * (x.T @ gradient + np.float32(0.0001) * w)
    b -= rate * gradient.sum(axis=0)


def train_source(x, y, seed):
    rng = np.random.default_rng(seed)
    w = np.zeros((x.shape[1], CLASSES), dtype=np.float32)
    b = np.zeros(CLASSES, dtype=np.float32)
    for _ in range(SOURCE_EPOCHS):
        order = rng.permutation(len(y))
        for start in range(0, len(order), SOURCE_BATCH):
            idx = order[start:start + SOURCE_BATCH]
            gradient_step(x[idx], y[idx], w, b, np.float32(0.2))
    return w, b


def update_model(w, b, source_x, source_y, selected_x, selected_y, seed, round_number):
    """Replay an update from the parent checkpoint and the ordered label snapshot."""
    w, b = w.copy(), b.copy()
    rng = np.random.default_rng(np.random.SeedSequence([seed, round_number, 100]))
    for _ in range(ADAPT_STEPS):
        src = rng.integers(len(source_y), size=ADAPT_BATCH_PER_DOMAIN)
        tgt = rng.integers(len(selected_y), size=ADAPT_BATCH_PER_DOMAIN)
        x = np.concatenate((source_x[src], selected_x[tgt]))
        y = np.concatenate((source_y[src], selected_y[tgt]))
        gradient_step(x, y, w, b, np.float32(0.04))
    return w, b


def select_batch(method, candidates, features, w, b, count, rng):
    if method == "random":
        return rng.choice(candidates, size=count, replace=False)
    p = probabilities(features[candidates], w, b)
    entropy = -(p * np.log(np.maximum(p, np.finfo(p.dtype).tiny))).sum(axis=1)
    return candidates[np.lexsort((candidates, -entropy))[:count]]


def rows_id(prefix, index):
    return f"{prefix}{index + 1:06d}"


def save_checkpoint(directory, model, parent, w, b, rows):
    path = directory / f"{model}.npz"
    np.savez_compressed(path, w=w, b=b)
    rows.append(dict(model=model, parent=parent, file=path.name,
                     parameters_sha256=hashlib.sha256(w.tobytes() + b.tobytes()).hexdigest(),
                     file_sha256=sha256(path)))


def run(cache, out):
    begin = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = cache / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected in FEATURES.items():
        if sha256(cache / filename) != expected:
            raise ValueError(f"Input hash mismatch: {filename}")
    sx, sy = load_domain(cache / "Product_Product.csv")
    tx, ty = load_domain(cache / "Product_RealWorld.csv")
    seen = set()
    source_rows, source_excluded = unique_features(sx, seen)
    target_rows, target_excluded = unique_features(tx, seen)
    train, val = split_rows(sy, source_rows)
    pool, test = split_rows(ty, target_rows)
    if len(set(ty[pool])) != CLASSES or len(set(ty[test])) != CLASSES:
        raise ValueError("Incomplete class coverage after splitting")
    split_rows_export = []
    for prefix, y, partitions in [
        ("Pr", sy, [(train, "source_train"), (val, "source_val"), (source_excluded, "excluded")]),
        ("Rw", ty, [(pool, "target_pool"), (test, "target_test"), (target_excluded, "excluded")]),
    ]:
        for indices, partition in partitions:
            for i in indices:
                split_rows_export.append(dict(sample_id=rows_id(prefix, i), split=partition,
                                              class_id=int(y[i]), row_1based=int(i + 1)))
    write_table(out / "splits.csv", split_rows_export)
    source_x, source_y = sx[train], sy[train]
    # All selectors and updates receive only target-pool features.
    px = tx[pool]
    trajectories, checkpoints, queries, top3, registrations = [], [], [], [], []
    source_validations = []
    for seed in SEEDS:
        started = time.perf_counter()
        source_w, source_b = train_source(source_x, source_y, seed)
        elapsed = time.perf_counter() - started
        source_key = f"s{seed}-source"
        save_checkpoint(checkpoint_dir, source_key, "", source_w, source_b, checkpoints)
        source_validations.append(dict(seed=seed, source_val_size=len(val),
            source_val_top1=float(np.mean(np.argmax(sx[val] @ source_w + source_b, axis=1) == sy[val])),
            source_train_seconds=elapsed))
        if seed == SEEDS[0]:
            ranking = np.argsort(-(px @ source_w + source_b), axis=1, kind="stable")[:, :3]
            for pool_i, ranks in zip(pool, ranking):
                top3.append(dict(sample_id=rows_id("Rw", pool_i), seed=seed,
                                 rank1=int(ranks[0]), rank2=int(ranks[1]), rank3=int(ranks[2]),
                                 reference_label=int(ty[pool_i])))
        for method in METHODS:
            w, b = source_w, source_b
            previous = source_key
            sample_order = []
            query_rng = np.random.default_rng(np.random.SeedSequence([seed, 200]))
            registrations.append((seed, method, 0, source_key, 0))
            for r in range(1, 6):
                candidates = np.setdiff1d(np.arange(len(pool)), sample_order)
                expected = r * len(pool) // 100
                start_query = time.perf_counter()
                selected = select_batch(method, candidates, px, w, b,
                                        expected - len(sample_order), query_rng)
                query_seconds = time.perf_counter() - start_query
                sample_order.extend(selected.tolist())
                key = f"s{seed}-{method}-r{r}"
                for rank, i in enumerate(selected, 1):
                    queries.append(dict(seed=seed, method=method, round=r, rank=rank,
                        sample_id=rows_id("Rw", pool[i]), oracle_label=int(ty[pool[i]]),
                        pre_model=previous, post_model=key))
                # The simulated oracle returns only labels of accepted rows.
                revealed_labels = ty[pool[sample_order]]
                start_update = time.perf_counter()
                w, b = update_model(w, b, source_x, source_y, px[sample_order],
                                    revealed_labels, seed, r)
                update_seconds = time.perf_counter() - start_update
                save_checkpoint(checkpoint_dir, key, previous, w, b, checkpoints)
                trajectories.append(dict(seed=seed, method=method, round=r,
                    added_labels=len(selected), unique_labels=len(sample_order),
                    query_seconds=query_seconds, update_seconds=update_seconds, model=key))
                registrations.append((seed, method, r, key, expected))
                previous = key
            print(f"Finished training {seed}/{method}", flush=True)
    # Only after the last training trajectory completes, independently score frozen checkpoints.
    metrics, predictions = [], []
    counts = np.bincount(ty[test], minlength=CLASSES)
    for seed, method, r, key, budget in registrations:
        with np.load(checkpoint_dir / f"{key}.npz") as checkpoint:
            pred = np.argmax(tx[test] @ checkpoint["w"] + checkpoint["b"], axis=1)
        match = pred == ty[test]
        correct_by_class = np.bincount(ty[test][match], minlength=CLASSES)
        metrics.append(dict(seed=seed, method=method, round=r, unique_labels=budget,
            budget_fraction=budget / len(pool), top1=float(np.mean(match)),
            macro_recall=float(np.mean(correct_by_class / counts)), correct=int(match.sum()),
            test_size=len(test), model=key))
        for i, guess in zip(test, pred):
            predictions.append(dict(seed=seed, method=method, round=r, sample_id=rows_id("Rw", i),
                                    reference_label=int(ty[i]), prediction=int(guess), model=key))
    for name, records in [("metrics", metrics), ("checkpoints", checkpoints), ("queries", queries),
                          ("source_pool_top3", top3), ("source_validation", source_validations),
                          ("predictions", predictions), ("timing", trajectories)]:
        write_table(out / f"{name}.csv", records)
    run_info = dict(source_repository="https://github.com/SJTUjinmao/Office-Home-Features",
        revision="d722485d101b5d81309561ffeb3519fd5b5e274a", file_sha256=FEATURES,
        raw_rows=dict(source=len(sx), target=len(tx)),
        normalized_feature_duplicate_exclusions=dict(source=len(source_excluded), target=len(target_excluded)),
        split_counts=dict(source_train=len(train), source_val=len(val), target_pool=len(pool), target_test=len(test)),
        num_classes=CLASSES, features=2048, split_seed=SPLIT_SEED, paired_seeds=SEEDS,
        methods=METHODS, source_epochs=SOURCE_EPOCHS, source_batch=SOURCE_BATCH,
        source_lr=0.2, adapt_steps_per_round=ADAPT_STEPS, adapt_n_per_domain_per_step=ADAPT_BATCH_PER_DOMAIN,
        adapt_lr=0.04, weight_decay=0.0001, feature_l2_norm=10,
        numpy=np.__version__, python=platform.python_version(), platform=platform.platform(),
        processor=platform.processor(), device="CPU", blas_threads=1, seconds=time.perf_counter()-begin,
        script_sha256=sha256(Path(__file__)), split_sha256=sha256(out/"splits.csv"))
    (out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run_info, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "build/model_pilot_data")
    parser.add_argument("--out", type=Path, default=ROOT / "results/model_pilot")
    args = parser.parse_args()
    run(args.cache, args.out)
