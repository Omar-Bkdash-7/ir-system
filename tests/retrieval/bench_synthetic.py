"""Old vs new retrieval on a synthetic Zipf corpus (no real data needed). Run from the project root:

    python -m tests.retrieval.bench_synthetic                  # 100,000 documents
    python -m tests.retrieval.bench_synthetic --docs 523000    # Quora-sized (needs several GB of RAM and a few minutes)

The corpus is written in the real on-disk formats; each variant then runs in its OWN process so load time and memory are
clean numbers. 'old' = the code this rewrite replaced (tests/retrieval/legacy), called the way the service called it
(top_k=50 became candidates_k=1000). 'new' = services.retrieval.core. Numbers depend on the machine: compare old vs new,
not absolute values.
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = "bench"


def rss_mb() -> float:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def make_queries(count, vocab, seed=11):
    """5-term queries; term ranks log-uniform in [1, 5000]: common and rare words mixed."""
    rng = np.random.default_rng(seed)
    return [[f"w{min(vocab - 1, int(np.exp(rng.uniform(0, np.log(5000)))) - 1)}" for _ in range(5)] for _ in range(count)]


# ------------------------------------------------------------------------------------------------ corpus generation
def generate(workdir: Path, docs: int, vocab: int, avg_len: int) -> None:
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer

    base = workdir / "data" / "indices" / DATASET
    if (base / "tfidf" / "tfidf_doc_ids.json").exists():
        print(f"corpus already generated in {workdir}")
        return
    shutil.rmtree(workdir, ignore_errors=True)
    for kind in ("bm25", "tfidf"):
        (base / kind).mkdir(parents=True)
    started = time.perf_counter()
    rng = np.random.default_rng(0)
    p = 1.0 / np.arange(1, vocab + 1) ** 1.05
    p /= p.sum()
    lengths = np.maximum(5, rng.poisson(avg_len, docs))
    terms = rng.choice(vocab, size=int(lengths.sum()), p=p).astype(np.int64)
    doc_of = np.repeat(np.arange(docs, dtype=np.int64), lengths)
    unique, tf = np.unique(terms * docs + doc_of, return_counts=True)
    term_idx, doc_idx = unique // docs, unique % docs
    doc_ids = [f"D{i}" for i in range(docs)]
    doc_ids_arr = np.array(doc_ids, dtype=object)
    boundaries = np.searchsorted(term_idx, np.arange(vocab + 1))
    inverted = {}
    for t in range(vocab):
        s, e = boundaries[t], boundaries[t + 1]
        if e > s:
            inverted[f"w{t}"] = list(zip(doc_ids_arr[doc_idx[s:e]].tolist(), tf[s:e].tolist()))
    df = {term: len(entries) for term, entries in inverted.items()}
    idf = {term: float(np.log((docs - d + 0.5) / (d + 0.5) + 1.0)) for term, d in df.items()}
    data = {"inverted_index": inverted, "doc_frequencies": df, "idf_scores": idf,
            "doc_lengths": {doc_ids[i]: int(lengths[i]) for i in range(docs)}, "avg_doc_length": float(lengths.mean()),
            "total_docs": docs, "doc_ids": doc_ids}
    with open(base / "bm25" / "bm25_index.pkl", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"BM25 index: {len(inverted):,} terms, {len(unique):,} postings ({time.perf_counter() - started:.0f}s)")
    del data, inverted

    words = np.array([f"w{t}" for t in range(vocab)], dtype=object)
    texts = [" ".join(chunk) for chunk in np.split(words[terms], np.cumsum(lengths)[:-1])]
    vectorizer = TfidfVectorizer(sublinear_tf=True, norm="l2")
    matrix = vectorizer.fit_transform(texts)
    sp.save_npz(base / "tfidf" / "tfidf_matrix.npz", matrix)
    with open(base / "tfidf" / "tfidf_vectorizer.pkl", "wb") as handle:
        pickle.dump(vectorizer, handle)
    with open(base / "tfidf" / "tfidf_doc_ids.json", "w") as handle:
        json.dump(doc_ids, handle)
    print(f"TF-IDF index: {matrix.shape}, {matrix.nnz:,} non-zeros ({time.perf_counter() - started:.0f}s)")


# ------------------------------------------------------------------------------------------------ one variant, one process
def run_child(mode: str, workdir: Path, queries: int, vocab: int) -> dict:
    os.chdir(workdir)
    sys.path.insert(0, str(ROOT))
    import warnings
    warnings.simplefilter("ignore")
    kind = "bm25" if "bm25" in mode else "tfidf"
    qs = make_queries(queries, vocab)
    rss_before = rss_mb()
    started = time.perf_counter()
    if mode.startswith("old"):
        stub = types.ModuleType("services.preprocessing.preprocessing_service")
        stub.preprocess = lambda text, mode=None, config=None: text.split()
        sys.modules["services.preprocessing.preprocessing_service"] = stub
        import importlib
        legacy = importlib.import_module(f"tests.retrieval.legacy.{kind}_retriever")
        retriever = (legacy.BM25Retriever if kind == "bm25" else legacy.TFIDFRetriever)(DATASET)
        retriever.load()
        run = lambda tokens: retriever.retrieve(tokens, top_k=1000)  # noqa: E731  (50 * 20: what the service asked for)
    else:
        from services.retrieval.core import indexes as I
        from services.retrieval.core.compact import CompactCache
        from services.retrieval.core.query import QueryContext
        index = I.load_index(kind, DATASET, embedding_mmap=True, cache=CompactCache(workdir / "cache"))
        options = I.SearchOptions()

        def run(tokens):
            context = QueryContext(text="", tokens=tokens, encoder=None, preprocess=str.split)
            return index.search(index.prepare(context), 50, options)
    load_s = time.perf_counter() - started
    run(qs[0])  # first query: caches, page faults
    times = []
    for tokens in qs:
        t0 = time.perf_counter()
        run(tokens)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {"mode": mode, "load_s": round(load_s, 2), "rss_mb": round(rss_mb() - rss_before),
            "median_ms": round(float(np.median(times)), 2), "p95_ms": round(float(np.percentile(times, 95)), 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=int, default=100_000)
    parser.add_argument("--vocab", type=int, default=30_000)
    parser.add_argument("--avg-len", type=int, default=50)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--workdir", default="/tmp/retrieval-bench")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    workdir = Path(args.workdir)
    if args.child:
        print(json.dumps(run_child(args.child, workdir, args.queries, args.vocab)))
        return
    generate(workdir, args.docs, args.vocab, args.avg_len)
    results = []
    for mode in ("old-bm25", "new-bm25-cold", "new-bm25-warm", "old-tfidf", "new-tfidf-cold", "new-tfidf-warm"):
        if mode.endswith("cold"):
            shutil.rmtree(workdir / "cache", ignore_errors=True)
        out = subprocess.run([sys.executable, "-m", "tests.retrieval.bench_synthetic", "--child", mode, "--workdir", str(workdir),
                              "--queries", str(args.queries), "--vocab", str(args.vocab)],
                             cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
        if out.returncode != 0:
            print(out.stderr[-2000:])
            raise SystemExit(f"{mode} failed")
        results.append(json.loads(out.stdout.strip().splitlines()[-1]))
    print(f"\ncorpus: {args.docs:,} documents, vocabulary {args.vocab:,}; {args.queries} queries of 5 terms (common and rare mixed)\n")
    print(f"{'variant':<16}{'load (s)':>10}{'extra RSS (MB)':>16}{'median (ms)':>13}{'p95 (ms)':>10}")
    for r in results:
        print(f"{r['mode']:<16}{r['load_s']:>10}{r['rss_mb']:>16}{r['median_ms']:>13}{r['p95_ms']:>10}")
    old = {r["mode"][4:]: r for r in results if r["mode"].startswith("old")}
    for kind in ("bm25", "tfidf"):
        new = next(r for r in results if r["mode"] == f"new-{kind}-warm")
        print(f"{kind}: {old[kind]['median_ms'] / new['median_ms']:.0f}x faster per query (median), "
              f"{old[kind]['rss_mb'] / max(1, new['rss_mb']):.1f}x less memory, "
              f"{old[kind]['load_s'] / max(0.001, new['load_s']):.0f}x faster start from the cache")


if __name__ == "__main__":
    main()
