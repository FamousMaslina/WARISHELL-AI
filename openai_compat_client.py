from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import httpx
from rich.live import Live
from rich.text import Text

from console_utils import console


class OpenAICompatClient:
    """
    Client for OpenAI API-compatible endpoints.
    
    Supports any endpoint that follows the OpenAI chat completion API format,
    including official OpenAI, vLLM, LM Studio, LocalAI, and other compatible servers.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_s: int = 1800,
        default_ctx: Optional[int] = None,
        use_custom_params: bool = False,
    ):
        self.base = base_url.rstrip("/")
        self.use_custom_params = use_custom_params
        
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
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
            headers=headers,
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
        """
        Send a chat completion request to the endpoint.
        
        Args:
            model: The model identifier to use
            messages: List of message dicts with 'role' and 'content' keys
            options: Optional generation parameters (temperature, top_p, etc.)
            stream: Whether to stream the response (always True for timeout handling)
            keep_alive: Ignored for remote endpoints
        
        Returns:
            The generated response content as a string (includes reasoning content if available)
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,  # Always stream to reset timeout on token activity
        }
        
        # Only include custom params if use_custom_params is True
        if self.use_custom_params and options:
            # Map options to OpenAI-compatible format
            if "temperature" in options:
                payload["temperature"] = options["temperature"]
            if "top_p" in options:
                payload["top_p"] = options["top_p"]
            if "top_k" in options:
                # Some endpoints support top_k directly
                payload["top_k"] = options["top_k"]
            if "repetition_penalty" in options:
                payload["repetition_penalty"] = options["repetition_penalty"]
            if "max_tokens" in options:
                payload["max_tokens"] = options["max_tokens"]
            if "stop" in options:
                payload["stop"] = options["stop"]
            # Support reasoning_effort for OpenAI reasoning models (o1, o3, etc.)
            if "reasoning_effort" in options:
                payload["reasoning_effort"] = options["reasoning_effort"]
        
        # Add max_tokens if not already set.
        # qwen3.5 thought itd be a good idea to cap at at 4096
        if self.default_ctx is not None and "max_tokens" not in payload:
            #output_budget = max(8192, int(self.default_ctx * 0.80))
            payload["max_tokens"] = min(self.default_ctx, 65536)

        # Stream the response, showing a live elapsed-time indicator while waiting.
        done_received = False
        finish_reason: Optional[str] = None
        choices_data: List[Dict[str, Any]] = []

        # 'thinking' = model is in <think> / reasoning phase
        # 'generating' = regular content tokens are arriving
        phase = "thinking"

        def _status_line(elapsed: float) -> Text:
            mins, secs = divmod(int(elapsed), 60)
            ts = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
            line = Text()
            if phase == "thinking":
                line.append("⚙ Thinking", style="cyan")
            else:
                line.append("✍ Generating", style="green")
            line.append(f"  [{ts}]", style="dim")
            return line

        start = time.monotonic()
        with Live(_status_line(0), console=console, refresh_per_second=4, transient=True) as live:
            async with self.client.stream("POST", "/chat/completions", json=payload) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        done_received = True
                        break
                    try:
                        data = json.loads(line)
                        choices_data.append(data)
                        for choice in data.get("choices") or []:
                            fr = choice.get("finish_reason")
                            if fr:
                                finish_reason = fr
                            # Switch label once regular content tokens start arriving
                            delta = choice.get("delta") or {}
                            if delta.get("content"):
                                phase = "generating"
                    except json.JSONDecodeError:
                        pass
                    live.update(_status_line(time.monotonic() - start))

        elapsed = time.monotonic() - start
        mins, secs = divmod(int(elapsed), 60)
        ts = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
        console.print(f"  [dim]Done in {ts}[/]")

        # Warn on truncation so the engine log makes the root cause obvious.
        if not done_received:
            console.print("[yellow]Warning: OpenAICompat stream ended without [DONE] signal[/]")
        if finish_reason == "length":
            console.print(
                f"[red]Warning: OpenAICompat response truncated (finish_reason=length). "
                f"max_tokens={payload.get('max_tokens')}. "
                "Increase WARISHELL_CONTEXT_WINDOW or pass a larger --context-window.[/]"
            )

        # Reconstruct the full response from streaming chunks
        # This mimics the non-streaming response format
        full_message: Dict[str, str] = {"content": "", "reasoning_content": ""}
        
        for chunk_data in choices_data:
            choices = chunk_data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or choices[0].get("message") or {}
            
            # Accumulate reasoning content
            if "reasoning_content" in delta and delta["reasoning_content"]:
                full_message["reasoning_content"] += delta["reasoning_content"]
            
            # Accumulate regular content
            if "content" in delta and delta["content"] is not None:
                full_message["content"] += delta["content"]
        
        # Combine reasoning and regular content, wrapping reasoning in tags
        # so the parser can extract it properly
        reasoning_content = full_message.get("reasoning_content", "")
        regular_content = full_message.get("content", "")
        
        if reasoning_content:
            if regular_content:
                return "<</think>" + reasoning_content + "\n\n" + regular_content + ""
            else:
                return "<</think>" + reasoning_content
        
        return regular_content or ""

    async def unload(self, model: str) -> None:
        """
        Unload a model from the endpoint.
        
        For remote endpoints, this is typically a no-op as models are
        managed by the server. Some endpoints may support this via
        a separate API.
        """
        # Most OpenAI-compatible endpoints don't support model unloading
        # via the standard API, so this is a no-op by default
        pass

    async def stop_all(self) -> None:
        """
        Stop all running completions.
        
        For remote endpoints, this is typically a no-op.
        """
        pass