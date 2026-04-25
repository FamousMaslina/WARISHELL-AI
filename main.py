#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from assets import ensure_flag_assets
from config import DEFAULT_MODELS, MODEL_CONTEXT_WINDOW, names_from_models
from console_utils import console, setup_console_tee
from decisions import ModelDecision
from engine import Engine, attach_agents, seed_world
from resume import attach_agents_from_map, find_resume_point, load_memories, load_world_snapshot
from pydantic import ValidationError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama multi‑LLM geopolitics simulator")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Ollama model names (space‑separated). Will assign 1:1 to countries in order.")
    parser.add_argument("--countries", nargs="*", default=None, help="Optional country names (defaults to model count).")
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("runs/demo"))
    parser.add_argument("--ollama", type=str, default="http://localhost:11434")
    parser.add_argument(
        "--openrouter",
        type=str,
        default=os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter base URL (default set via OPENROUTER_URL/https://openrouter.ai/api/v1).",
    )
    parser.add_argument(
        "--openrouter-key",
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (can also be provided via OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--openai_compat_url",
        type=str,
        default=os.environ.get("OPENAI_COMPAT_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible endpoint base URL (default set via OPENAI_COMPAT_URL).",
    )
    parser.add_argument(
        "--openai_compat_key",
        type=str,
        default=os.environ.get("OPENAI_COMPAT_API_KEY"),
        help="OpenAI-compatible endpoint API key (optional, can also be provided via OPENAI_COMPAT_API_KEY).",
    )
    parser.add_argument(
        "--openai_compat_use_custom_params",
        action="store_true",
        default=False,
        help="Enable custom generation params (temp, top_p, etc.) for OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Override the shared LLM context window (num_ctx/max_context_tokens).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


async def _wrap_ask(eng: Engine):
    orig_ask = eng._ask_agent

    async def wrapped(cname, agent):
        cname, raw, think, parsed = await orig_ask(cname, agent)
        if parsed is not None:
            try:
                dec = ModelDecision.model_validate(parsed)
                eng._last_decisions[cname] = (dec, raw, think)
            except ValidationError:
                pass
        return cname, raw, think, parsed

    eng._ask_agent = wrapped  # type: ignore[attr-defined]


async def main_async() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    setup_console_tee(out_dir)
    resume_info = find_resume_point(out_dir)
    context_window = args.context_window or MODEL_CONTEXT_WINDOW

    def ensure_openrouter_configured(agent_list) -> None:
        needing_openrouter = [a for a in agent_list if a.provider == "or"]
        if needing_openrouter and not args.openrouter_key:
            info = ", ".join(f"{a.country} ({a.model})" for a in needing_openrouter)
            console.print(f"[red]Error:[/] The following models are tagged for OpenRouter in config.py: {info}")
            console.print("Provide --openrouter-key or set OPENROUTER_API_KEY to continue.")
            sys.exit(1)

    def ensure_openai_compat_configured(agent_list) -> None:
        needing_openai_compat = [a for a in agent_list if a.provider == "openai_compat"]
        if needing_openai_compat:
            info = ", ".join(f"{a.country} ({a.model})" for a in needing_openai_compat)
            console.print(f"[yellow]Info:[/] The following models use OpenAI-compatible endpoint: {info}")
            console.print("Ensure --openai_compat_url is set correctly. API key is optional.")

    if resume_info:
        last_turn, snap_path = resume_info
        world, model_map = load_world_snapshot(snap_path)
        world.news = []
        agents = attach_agents_from_map(world, model_map, args.models[: len(world.countries)])
        ensure_openrouter_configured(agents)
        ensure_openai_compat_configured(agents)
        eng = Engine(
            world,
            agents,
            out_dir,
            args.ollama,
            args.openrouter,
            args.openrouter_key,
            args.openai_compat_url,
            args.openai_compat_key,
            args.openai_compat_use_custom_params,
            context_window,
            seed=args.seed,
        )
        ensure_flag_assets(out_dir, agents)
        load_memories(out_dir / "history", agents, cap=Engine.MAX_PRIVATE_MEM_LINES)
        console.print(f"[cyan]Resuming[/] from turn {last_turn} in {out_dir}. Starting at turn {last_turn + 1} for {args.turns} additional turn(s).")
        try:
            eng.io.save_json(out_dir / "rt.json", eng._world_snapshot())
        except Exception:
            pass
    else:
        if args.countries:
            names = args.countries
        else:
            names = names_from_models(args.models)
        if len(names) > len(args.models):
            console.print("[red]Error:[/] Provide at least as many models as countries.")
            sys.exit(1)

        world = seed_world(names[: len(args.models)])
        world.turn = 0
        agents = attach_agents(world, args.models[: len(names)])
        ensure_openrouter_configured(agents)
        ensure_openai_compat_configured(agents)

        if out_dir.exists() and any(out_dir.iterdir()):
            console.print(f"[yellow]Warning:[/] output dir {out_dir} is not empty; data may be overwritten.")
        eng = Engine(
            world,
            agents,
            out_dir,
            args.ollama,
            args.openrouter,
            args.openrouter_key,
            args.openai_compat_url,
            args.openai_compat_key,
            args.openai_compat_use_custom_params,
            context_window,
            seed=args.seed,
        )

        turn0 = eng.io.turn_dir(0)
        eng.io.save_json(turn0 / "world.json", eng._world_snapshot())
        ensure_flag_assets(out_dir, agents)
        manifest = [{
            "nation": a.country,
            "model_id": a.model,
            "model_alias": a.alias,
            "flag": (Path(a.flag).name if a.flag else None),
            "flag_url": (f"flags/{Path(a.flag).name}" if a.flag else None),
        } for a in agents]
        eng.io.save_json(out_dir / "agents_manifest.json", {"turn": 0, "mapping": manifest})
        console.print("[cyan]Agents manifest[/]:")
        for row in manifest:
            console.print(f"  {row['nation']}: {row['model_alias']} ({row['model_id']})")
        try:
            eng.io.save_json(out_dir / "rt.json", eng._world_snapshot())
        except Exception:
            pass

    eng._last_decisions = {}
    await _wrap_ask(eng)

    try:
        await eng.run(args.turns)
    finally:
        await eng.close()


def main() -> None:
    if os.name != "nt":
        try:
            import uvloop  # type: ignore
            uvloop.install()
        except Exception:
            pass
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
