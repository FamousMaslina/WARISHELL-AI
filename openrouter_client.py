from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx

from console_utils import console


class OpenRouterClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: int = 1800,
        default_ctx: Optional[int] = None,
    ):
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        self.base = base_url.rstrip("/")
        # Use timeout with separate read timeout that resets on token activity
        # The read timeout is set high to allow slow token generation
        self.client = httpx.AsyncClient(
            base_url=self.base,
            timeout=httpx.Timeout(
                connect=10.0,      # Time to establish connection
                read=timeout_s,    # Time between tokens (resets on each token)
                write=30.0,        # Time to send request
                pool=10.0,         # Time to get connection from pool
            ),
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
        stream: bool = True,  # Always use streaming for timeout reset on token activity
        keep_alive: Optional[Any] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,  # Always stream to reset timeout on token activity
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
                # Use stream=True to get streaming response
                resp = await self.client.post("/chat/completions", json=payload, stream=True)
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 429 and attempts == 0:
                    console.print("[yellow]OpenRouter rate limit[/]: waiting 10s before retrying...")
                    attempts += 1
                    await asyncio.sleep(10)
                    continue
                raise

        # Stream the response - each token received resets the read timeout
        choices_data: List[Dict[str, Any]] = []
        done_received = False
        async for line in resp.aiter_lines():
            if not line:
                continue
            # Skip SSE prefix if present
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                done_received = True
                break
            try:
                data = json.loads(line)
                choices_data.append(data)
            except json.JSONDecodeError:
                pass

        # Verify we got the complete response
        if not done_received:
            console.print("[yellow]Warning: OpenRouter stream ended without [DONE] signal[/]")

        # Reconstruct the full response from streaming chunks
        full_message: Dict[str, str] = {"content": ""}
        
        for chunk_data in choices_data:
            choices = chunk_data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            
            # Accumulate content
            if "content" in delta and delta["content"]:
                full_message["content"] += delta["content"]

        return full_message.get("content", "") or ""

    async def unload(self, model: str) -> None:
        return

    async def stop_all(self) -> None:
        return
