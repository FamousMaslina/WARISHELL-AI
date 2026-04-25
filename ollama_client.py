from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from console_utils import console
import httpx


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: int = 1800, default_ctx: int = 8192):
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
            )
        )
        # Optional default context window for all calls (maps to Ollama's num_ctx option)
        self.default_ctx = default_ctx

    async def close(self):
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
        keep_alive: Optional[Any] = 0,
    ) -> str:
        base_opts: Dict[str, Any] = dict(options) if options else {}
        if self.default_ctx is not None:
            base_opts.setdefault("num_ctx", self.default_ctx)

        def _build_payload(opt_dict: Dict[str, Any]) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": True,  # Always stream to reset timeout on token activity
            }
            if opt_dict:
                payload["options"] = opt_dict
            if keep_alive is not None:
                payload["keep_alive"] = keep_alive
            return payload

        attempts: List[Dict[str, Any]] = [dict(base_opts)]
        ctx_val = base_opts.get("num_ctx")
        if ctx_val:
            # Gracefully fall back for models that cannot handle large context windows.
            for fb in (8192, 4096):
                if ctx_val > fb:
                    fb_opts = dict(base_opts)
                    fb_opts["num_ctx"] = fb
                    attempts.append(fb_opts)

        last_exc: Optional[httpx.HTTPStatusError] = None
        error_notes: List[str] = []

        for opt_dict in attempts:
            try:
                # Use stream=True to get streaming response
                r = await self.client.post("/api/chat", json=_build_payload(opt_dict), stream=True)
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                last_exc = e
                resp_txt = ""
                if e.response is not None:
                    try:
                        resp_txt = e.response.text
                    except Exception:
                        resp_txt = ""
                note = f"Attempt with num_ctx={opt_dict.get('num_ctx')} failed: {e}"
                if resp_txt.strip():
                    note = f"{note} | body: {resp_txt.strip()}"
                error_notes.append(note)
                continue

            # Stream the response - each token received resets the read timeout
            chunks: List[str] = []
            done_received = False
            try:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # Check for error in response
                        if "error" in obj:
                            raise httpx.HTTPStatusError(
                                f"Ollama error: {obj['error']}",
                                request=r.request,
                                response=r,
                            )
                        # Check for end-of-stream signal
                        if obj.get("done") is True:
                            done_received = True
                            break
                        content = obj.get("message", {}).get("content")
                        if content:
                            chunks.append(content)
                    except json.JSONDecodeError:
                        # Skip lines that aren't valid JSON
                        pass
            except httpx.ReadTimeout:
                # Timeout during streaming - this is expected for slow models
                # but should be rare with the high read timeout
                raise

            # Verify we got the complete response
            if not done_received:
                console.print("[yellow]Warning: Ollama stream ended without 'done: true' signal[/]")
            
            return "".join(chunks)

        if last_exc:
            for note in error_notes:
                try:
                    last_exc.add_note(note)
                except Exception:
                    pass
            raise last_exc

        # Should never reach here, but satisfy type checkers.
        raise RuntimeError("Ollama chat failed without raising an HTTP error.")

    async def unload(self, model: str) -> None:
        """Ask Ollama to unload a model immediately (free VRAM)."""
        # Newer servers: DELETE /api/ps/{name}
        try:
            r = await self.client.delete(f"/api/ps/{model}")
            r.raise_for_status()
            return
        except Exception:
            pass

        # Also try the documented stop endpoint (name).
        try:
            r = await self.client.post("/api/stop", json={"name": model})
            r.raise_for_status()
            return
        except Exception:
            pass

        # Fallback for servers that do not support /api/stop yet.
        try:
            r = await self.client.post("/api/generate", json={"model": model, "prompt": "", "stream": False, "keep_alive": 0})
            r.raise_for_status()
        except Exception:
            pass

    async def stop_all(self) -> None:
        """Best-effort: stop any running models before loading the next one."""
        try:
            r = await self.client.get("/api/ps")
            r.raise_for_status()
            data = r.json()
            models = data.get("models") or []
            for m in models:
                name = m.get("name") or m.get("model")
                if not name:
                    continue
                await self.unload(str(name))
        except Exception:
            pass
