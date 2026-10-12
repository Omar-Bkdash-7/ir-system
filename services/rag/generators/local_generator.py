"""Local CPU generator (transformers). torch/transformers are imported LAZILY so the service
starts fast and keeps working (remote-only) even if they are missing from the image.

Rules: load once (lazy, under a lock), never block the event loop (to_thread), run ONE inference
at a time, and always cap wall-clock time (generate(max_time=...)).
"""
from __future__ import annotations

import asyncio
import importlib.util
import time
from typing import Any, Dict, Optional, Tuple

from common.logging_config import setup_logging
from services.rag.config import RAGSettings
from services.rag.errors import GeneratorError
from services.rag.generators.base_generator import BaseGenerator, GenerationRequest, GenerationResult
from services.rag.local_model.manager import LocalModelManager
from services.rag.prompt_builder import build_messages, strip_reasoning

logger = setup_logging("rag.local_generator")


class LocalGenerator(BaseGenerator):
    source = "local"

    def __init__(self, settings: RAGSettings, manager: LocalModelManager):
        self.settings = settings
        self.manager = manager
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()

    # ---- availability -----------------------------------------------------------
    def unavailable_reason(self) -> Optional[str]:
        if not self.settings.local_enabled:
            return "Local model is disabled (RAG_LOCAL_ENABLED=false)."
        if self.manager.spec is None:
            return f"Unknown local model '{self.settings.local_model}'."
        state = self.manager.state
        if state not in ("ready", "downloading"):
            # A model copied by hand (docker cp) must be usable by the very next request:
            # bypass the status throttle with a forced (cheap) re-check of the files.
            self.manager.refresh_state(force=True)
            state = self.manager.state
        if state == "ready":
            return None
        if state == "downloading":
            return "The local model is still downloading."
        return "The local model is not downloaded yet."

    # ---- public API ---------------------------------------------------------------
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        reason = self.unavailable_reason()
        if reason:
            raise GeneratorError("local", "local_not_ready", reason)
        await self._ensure_loaded()
        started = time.perf_counter()
        async with self._infer_lock:  # CPU inference is serialized
            try:
                text, usage = await asyncio.to_thread(self._infer_sync, request)
            except GeneratorError:
                raise
            except Exception as exc:
                raise GeneratorError("local", "local_generation_failed", f"{type(exc).__name__}: {exc}"[:300]) from exc
        if not text:
            raise GeneratorError("local", "local_generation_failed", "The local model returned an empty answer.")
        return GenerationResult(
            text=text,
            source="local",
            model=self.manager.spec.repo_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            usage=usage,
        )

    # ---- loading ------------------------------------------------------------------
    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            try:
                self._model, self._tokenizer = await asyncio.to_thread(self._load_sync)
            except ImportError as exc:
                raise GeneratorError(
                    "local", "local_dependencies_missing", f"torch/transformers are not installed: {exc}",
                    hint="Add them to requirements/09-rag.txt and rebuild the rag image.",
                ) from exc
            except Exception as exc:
                raise GeneratorError("local", "local_load_failed", f"{type(exc).__name__}: {exc}"[:300]) from exc
            self.manager.loaded = True
            logger.info("Local model loaded into memory: %s", self.manager.spec.repo_id)

    def _load_sync(self) -> Tuple[Any, Any]:
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(self.settings.local_threads)
        path = str(self.manager.model_dir)
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.settings.local_dtype]
        # transformers >= 4.56 renamed torch_dtype -> dtype
        dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"
        kwargs: Dict[str, Any] = {"local_files_only": True, dtype_kw: dtype}
        if importlib.util.find_spec("accelerate") is not None:
            kwargs["low_cpu_mem_usage"] = True  # lowers peak RAM; requires `accelerate` on some versions
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        model.eval()
        return model, tokenizer

    # ---- inference ------------------------------------------------------------------
    def _infer_sync(self, request: GenerationRequest) -> Tuple[str, Dict[str, Any]]:
        import torch

        tok, model, cfg = self._tokenizer, self._model, request.config
        budget = self.settings.local_max_context_chars
        inputs = None
        for _ in range(3):  # shrink the context until the prompt fits the token budget
            messages = build_messages(
                request.query, request.documents, max_context_chars=budget,
                system_prompt=cfg.system_prompt, use_context=cfg.use_context,
            )
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
            if inputs["input_ids"].shape[-1] <= self.settings.local_max_input_tokens:
                break
            budget //= 2
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        sample = cfg.temperature > 0
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=min(cfg.max_tokens, self.settings.local_max_new_tokens),
            do_sample=sample,
            repetition_penalty=1.05,
            max_time=self.settings.local_max_seconds,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
        if sample:
            gen_kwargs.update(temperature=cfg.temperature, top_p=0.9)
        with torch.inference_mode():
            output = model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0, prompt_tokens:]  # only the newly generated part (no prompt echo)
        text = strip_reasoning(tok.decode(new_tokens, skip_special_tokens=True))
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": int(new_tokens.shape[-1])}
        return text, usage
