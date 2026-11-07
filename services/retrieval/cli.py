"""Command line helper for the retrieval service (run it from the project root / inside the container).

    python -m services.retrieval.cli status
    python -m services.retrieval.cli build-cache [--dataset cranfield ...] [--model bm25 tfidf]
    python -m services.retrieval.cli clear-cache [--dataset cranfield]

build-cache converts the BM25 / TF-IDF indices into the compact, memory-mappable form once, so the first search after a
deployment does not pay for it. It is optional: the service builds the same cache on first use.
"""
import argparse
import logging
import sys
import time

from services.retrieval.config import get_settings
from services.retrieval.core.compact import CompactCache, source_stamp, stamp_key
from services.retrieval.core.indexes import load_index, source_files
from services.retrieval.core.names import KINDS, LABELS, dataset_slug
from services.retrieval.core.store import IndexStore


def _cache_state(cache: CompactCache, dataset: str, kind: str) -> str:
    if kind == "embedding":
        return "n/a (memory-mapped directly)"
    try:
        key = stamp_key(source_stamp(source_files(kind, dataset)))
    except OSError:
        return "n/a"
    return "built" if cache.load(dataset_slug(dataset), kind, key) is not None else "missing"


def cmd_status(args: argparse.Namespace) -> int:
    settings = get_settings()
    cache = CompactCache(settings.cache_dir)
    store = IndexStore(cache=cache)
    print(f"compact cache directory: {settings.cache_dir}")
    for dataset in args.dataset or store.known_datasets():
        for kind in KINDS:
            built = store.is_built(dataset, kind)
            state = _cache_state(cache, dataset, kind) if built else "-"
            print(f"  {dataset:<24} {LABELS[kind]:<10} index: {'built' if built else 'not built':<10} compact cache: {state}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    settings = get_settings()
    cache = CompactCache(settings.cache_dir)
    store = IndexStore(cache=cache)
    failures = 0
    for dataset in args.dataset or store.known_datasets():
        for kind in args.model or KINDS:
            if not store.is_built(dataset, kind):
                continue
            started = time.perf_counter()
            try:
                index = load_index(kind, dataset, embedding_mmap=settings.embedding_mmap, cache=cache)
            except Exception as exc:
                failures += 1
                print(f"  {dataset} / {LABELS[kind]}: FAILED ({type(exc).__name__}: {exc})")
                continue
            print(f"  {dataset} / {LABELS[kind]}: ready in {time.perf_counter() - started:.1f}s  {index.stats()}")
    return 1 if failures else 0


def cmd_clear(args: argparse.Namespace) -> int:
    cache = CompactCache(get_settings().cache_dir)
    removed = cache.clear(dataset_slug(args.dataset) if args.dataset else None)
    print(f"removed {removed} cache folder(s) from {cache.root}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m services.retrieval.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (("status", cmd_status, "show indexes and compact caches"),
                                     ("build-cache", cmd_build, "pre-build the compact cache"),
                                     ("clear-cache", cmd_clear, "delete the compact cache")):
        command = sub.add_parser(name, help=help_text)
        command.set_defaults(handler=handler)
        if name == "clear-cache":
            command.add_argument("--dataset", help="only this dataset")
        else:
            command.add_argument("--dataset", nargs="*", help="dataset ids (default: every dataset found)")
        if name == "build-cache":
            command.add_argument("--model", nargs="*", choices=KINDS, help="models to build (default: all that exist)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
