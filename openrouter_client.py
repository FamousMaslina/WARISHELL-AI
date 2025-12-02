from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx

from console_utils import console


class OpenRouterClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: int = 600,
        default_ctx: Optional[int] = None,
    ):
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        self.base = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.base,
            timeout=timeout_s,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self.default_ctx = default_ctx

    async def close(self) -> None:
        try:
            await self.client.aclose()
        except Exception:
            pass

    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        keep_alive: Optional[Any] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if self.default_ctx is not None:
            payload.setdefault("max_context_tokens", self.default_ctx)
        if options:
            payload.update(options)

        attempts = 0
        while True:
            try:
                resp = await self.client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 429 and attempts == 0:
                    console.print("[yellow]OpenRouter rate limit[/]: waiting 10s before retrying...")
                    attempts += 1
                    await asyncio.sleep(10)
                    continue
                raise

        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content", "") or ""

    async def unload(self, model: str) -> None:
        return

    async def stop_all(self) -> None:
        return
