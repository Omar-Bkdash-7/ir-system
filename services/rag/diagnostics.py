"""Network self-diagnosis. Answers 'is it Docker DNS, the provider, or my token?' in one call.
Enabled only when RAG_ENABLE_DIAGNOSTICS=true. Never returns secrets."""
from __future__ import annotations

import asyncio
import socket
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from services.rag.config import RAGSettings
from services.rag.local_model.manager import LocalModelManager
from services.rag.netutil import has_dns_cause

HOSTS = [
    ("router.huggingface.co", "current inference endpoint"),
    ("huggingface.co", "model downloads (metadata)"),
    ("cas-bridge.xethub.hf.co", "model downloads (file storage)"),
    ("api-inference.huggingface.co", "LEGACY endpoint - expected to FAIL (retired by Hugging Face)"),
    ("google.com", "control: general internet DNS"),
]


async def _resolve(host: str) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        infos = await asyncio.wait_for(asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM), timeout=4)
        ips = sorted({i[4][0] for i in infos})[:4]
        return {"host": host, "ok": True, "addresses": ips, "ms": int((time.perf_counter() - started) * 1000)}
    except socket.gaierror as exc:
        return {"host": host, "ok": False, "errno": exc.errno, "error": exc.strerror, "ms": int((time.perf_counter() - started) * 1000)}
    except asyncio.TimeoutError:
        return {"host": host, "ok": False, "error": "DNS timeout (4s)", "ms": 4000}


async def _http_get(url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(6.0)) as client:
            response = await client.get(url, headers=headers or {})
        return {"url": url, "ok": response.status_code == 200, "status": response.status_code,
                "ms": int((time.perf_counter() - started) * 1000), "_response": response}
    except httpx.HTTPError as exc:
        return {"url": url, "ok": False, "error": type(exc).__name__, "ms": int((time.perf_counter() - started) * 1000)}


def _resolv_conf() -> List[str]:
    try:
        with open("/etc/resolv.conf", encoding="utf-8") as fh:
            return [ln.rstrip() for ln in fh.readlines()[:15]]
    except OSError:
        return []


async def run_network_diagnostics(settings: RAGSettings, manager: LocalModelManager) -> Dict[str, Any]:
    dns_results = await asyncio.gather(*(_resolve(h) for h, _ in HOSTS))
    for item, (_, note) in zip(dns_results, HOSTS):
        item["note"] = note

    router_catalog = await _http_get(f"{settings.remote_base_url}/models")
    response = router_catalog.pop("_response", None)
    if response is not None and response.status_code == 200:
        try:
            ids = {m.get("id") for m in response.json().get("data", [])}
            router_catalog["models_in_catalog"] = len(ids)
            router_catalog["configured_model_listed"] = settings.remote_model_id in ids
        except Exception:
            pass
    repo = manager.spec.repo_id if manager.spec else "Qwen/Qwen2.5-0.5B-Instruct"
    hub = await _http_get(f"https://huggingface.co/api/models/{repo}")
    hub.pop("_response", None)

    by_host = {r["host"]: r["ok"] for r in dns_results}
    if not by_host.get("google.com"):
        verdict = "The container has no working DNS/internet. Check Docker Desktop network, VPN or firewall."
    elif not by_host.get("router.huggingface.co"):
        verdict = "General DNS works but router.huggingface.co does not resolve: network/DNS filtering on this machine."
    elif not by_host.get("api-inference.huggingface.co"):
        verdict = "OK. Only the retired legacy host fails (expected). This is NOT a Docker DNS problem."
    else:
        verdict = "DNS is fine for all hosts."

    return {
        "verdict": verdict,
        "dns": dns_results,
        "resolv_conf": _resolv_conf(),
        "router_catalog": router_catalog,
        "hub_metadata": hub,
        "config": {
            "remote_base_url": settings.remote_base_url,
            "remote_model": settings.remote_model,
            "token_configured": settings.remote_configured,
            "local_model": settings.local_model,
            "models_dir": str(settings.models_dir),
        },
    }


async def run_download_probe(settings: RAGSettings, manager: LocalModelManager, sample_bytes: int = 2 * 1024 * 1024) -> Dict[str, Any]:
    """Downloads the first 2 MiB of the model through EXACTLY the path the real downloader uses (manifest API ->
    resolve -> redirect -> ranged GET) and reports speed. Answers 'is my network slow for this download?'."""
    cfg, spec = settings.download, manager.spec
    if spec is None:
        return {"ok": False, "verdict": "No local model is configured (RAG_LOCAL_MODEL)."}
    result: Dict[str, Any] = {"ok": False, "endpoint": cfg.hf_endpoint, "repo_id": spec.repo_id}
    timeout = httpx.Timeout(connect=cfg.connect_timeout, read=cfg.read_timeout, write=15.0, pool=15.0)

    async def probe() -> None:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            started = time.perf_counter()
            manifest = await client.get(f"{cfg.hf_endpoint}/api/models/{spec.repo_id}/revision/main", params={"blobs": "true"})
            result["manifest"] = {"status": manifest.status_code, "ms": int((time.perf_counter() - started) * 1000)}
            manifest.raise_for_status()
            data = manifest.json()
            weights = [f for f in data["siblings"] if f["rfilename"].endswith(".safetensors")]
            biggest = max(weights, key=lambda f: f["size"])
            url = f"{cfg.hf_endpoint}/{spec.repo_id}/resolve/{data['sha']}/{quote(biggest['rfilename'])}"
            started, received, first_byte = time.perf_counter(), 0, None
            headers = {"Range": f"bytes=0-{sample_bytes - 1}", "Accept-Encoding": "identity"}
            async with client.stream("GET", url, headers=headers) as response:
                async for chunk in response.aiter_raw(64 * 1024):
                    if first_byte is None:
                        first_byte = time.perf_counter() - started
                    received += len(chunk)
            elapsed = max(time.perf_counter() - started, 1e-6)
            result["file"] = {
                "name": biggest["rfilename"], "status": response.status_code, "ranged": response.status_code == 206,
                "final_host": response.url.host, "redirects": len(response.history), "bytes": received,
                "ttfb_ms": int(first_byte * 1000) if first_byte is not None else None,
                "mb_per_s": round(received / elapsed / 1e6, 2),
            }
            result["ok"] = response.status_code in (200, 206) and received > 0

    try:
        await asyncio.wait_for(probe(), timeout=60)
    except asyncio.TimeoutError:
        result["error"] = "The probe did not finish within 60 s: the connection is extremely slow or stalled."
    except httpx.HTTPError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:200]
        result["code"] = "dns_error" if has_dns_cause(exc) else "network_error"
    except (KeyError, ValueError) as exc:
        result["error"] = f"Unexpected API answer: {exc}"[:200]

    speed = (result.get("file") or {}).get("mb_per_s")
    if not result["ok"]:
        result["verdict"] = f"Cannot download from {cfg.hf_endpoint}: {result.get('error', 'no data received')}"
    elif speed is not None and speed < 0.3:
        result["verdict"] = (f"Reachable but SLOW ({speed} MB/s): the full model would take a long time. "
                             "Consider RAG_HF_FALLBACK_ENDPOINTS or a manual import (docs/RAG.md).")
    else:
        result["verdict"] = f"Download path OK ({speed} MB/s, final host {result['file']['final_host']})."
    return result
