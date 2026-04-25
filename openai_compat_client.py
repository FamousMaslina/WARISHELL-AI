from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


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
        timeout_s: int = 600,
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
        
        self.client = httpx.AsyncClient(
            base_url=self.base,
            timeout=timeout_s,
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
        stream: bool = False,
        keep_alive: Optional[Any] = None,
    ) -> str:
        """
        Send a chat completion request to the endpoint.
        
        Args:
            model: The model identifier to use
            messages: List of message dicts with 'role' and 'content' keys
            options: Optional generation parameters (temperature, top_p, etc.)
            stream: Whether to stream the response (not currently implemented)
            keep_alive: Ignored for remote endpoints
        
        Returns:
            The generated response content as a string
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
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
        
        # Add context window if configured (maps to max_tokens for some endpoints)
        if self.default_ctx is not None and "max_tokens" not in payload:
            # Use a reasonable default for max_tokens based on context window
            payload["max_tokens"] = min(self.default_ctx, 4096)
        
        resp = await self.client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content", "") or ""

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
