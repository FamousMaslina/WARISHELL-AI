# WARISHELL – AI Geopolitics Simulator

WARISHELL runs a multi‑agent, turn‑based geopolitics sandbox where each nation is piloted by an LLM (via Ollama or OpenRouter) or by a human. It wires strict JSON prompts into a rich economy/war engine, captures every turn to disk, and ships a zero‑install HTML viewer for live dashboards.

## What it does
- Spins up one agent per country and feeds them the exact same world state and system prompt each turn.
- Enforces a typed decision schema (trade, research, alliances, wars, loans, infrastructure, aid bids, puppet control, tax, policy).
- Models domestic politics: every nation has internal factions (military, business, workers, nationalists, greens, religious). Stability now comes from their approval and tension, forcing trade-offs instead of a flat stat.
- Resolves turn order: production → immediate actions → targeted/public market clearing → loans/alliances → wars → demographic/upkeep.
- Tracks memory per agent (recent news distilled into per‑player logs) and persists snapshots/inputs for auditing or resumes.
- Provides a File System Access–powered viewer (`viewer.html`) that tails `rt.json` or a run folder and renders wars, news, factions, puppetry, and balance sheets.

## Notes
This project was vibecoded with AI. It began as a dumb, fun idea: let a bunch of Qwens, Gemmas, and GPT-OSS models run countries and see who collapses first. Then it accidentally turned into something real. The engine expanded way too fast over 1–2 weeks, and suddenly the AIs were running a serious geopolitical simulator built by… other AIs.

The wild part is that the models actually take advantage of their conditions. Bigger OpenRouter models (see config.py) started thinking ahead. Even open 4B–30B models adapted to the resource constraints and political pressure. (side note-> Qwen3-4B-Thinking-2507 steamrolled Nonreasoning Mistral 3.1/3.2 24B.).

## Requirements
- Python 3.10+ with `httpx`, `pydantic`, `rich`, and `uvloop` (optional; skipped on Windows). Create a venv and `pip install httpx pydantic rich uvloop`.
- Running Ollama server reachable at `http://localhost:11434` (override with `--ollama`). Pull any models you reference in `--models` beforehand.
- Basic shell tools; no network calls are made beyond your Ollama endpoint.

## OpenRouter & Context
- Agents tagged with `provider: "or"` in `config.AGENT_PRESETS`, or prefixed with `or:` on the CLI (for example `--models or:gpt-4o-mini`), will be routed through OpenRouter instead of Ollama.
- Supply the OpenRouter endpoint via `--openrouter https://openrouter.ai/api/v1` and authentication via `--openrouter-key` (or `OPENROUTER_API_KEY`). OpenRouter usage without a key will fail early.
- The shared context window (`WARISHELL_CONTEXT_WINDOW` or `--context-window`) now drives Ollama's `num_ctx` and OpenRouter's `max_context_tokens`, ensuring both providers receive the same `ctx` budget.

## OpenAI-Compatible Endpoints
- Use any OpenAI API-compatible endpoint (e.g., vLLM, LM Studio, LocalAI, official OpenAI) by prefixing models with `openai_compat:` (e.g., `--models "openai_compat:meta-llama/Llama-3.1-8B-Instruct"`).
- Configure the endpoint with `--openai_compat_url` (default: `https://api.openai.com/v1`) and optional `--openai_compat_key` (or `OPENAI_COMPAT_API_KEY`).
- Enable custom generation parameters (temperature, top_p, etc.) with `--openai_compat_use_custom_params`. When disabled (default), the endpoint uses its own defaults.
- Example: `python main.py --models "openai_compat:llama-3.1-8b" --openai_compat_url "http://192.168.0.234:5559/v1" --context-window 131072 --seed 42`

## Quick start
```bash
cd /home/xxxx/Documents/warishell
# Optional: adjust presets in config.py (AGENT_PRESETS) or supply your own models/country names.
python main.py \
  --models hf.co/unsloth/Qwen3-4B-Thinking-2507-GGUF:Q4_K_M hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M \
  --turns 50 \
  --out runs/demo \
  --ollama http://localhost:11434
```
- Omit `--countries` to auto‑name nations from `config.AGENT_PRESETS` or the `Nation_X` fallback; pass `--countries` to pin names.
- Default output root is `runs/demo`; a warning is printed if it already contains files.

### Human player slots
Pass `PLAYER`, `PLAYER2`, … in `--models` (or in `AGENT_PRESETS`) to reserve interactive seats. When a human turn starts, the engine writes the exact system/user prompts into `turn_XXX/<NATION>__prompt_*.txt` and waits for you to drop a JSON response file (named `<PLAYER>_parsed.json`) before continuing.

## Run artifacts
- `turn_XXX/world.json` – full world snapshot per turn (countries, wars, factions, loans, puppets, news).
- `turn_XXX/<NATION>__raw.txt` / `__think.txt` / `__parsed.json` – model output, extracted `<think>` block, and validated actions.
- `agents_manifest.json` – model ↔ nation mapping plus flag filenames copied into `runs/.../flags/`.
- `rt.json` – most recent snapshot, refreshed every turn; the viewer polls this.
- `history/<NATION>.txt` – rolling private memory fed back to that agent (capped at 320 lines).

## Using the live viewer
1) Open `viewer.html` in a Chromium‑based browser.
2) Click “Open run folder” and pick the output directory (`runs/demo`) or “Open rt.json”.
3) Adjust poll interval if needed; pause/resume with the toolbar.
4) Flags are loaded from the `flags/` subfolder of the run; drop PNG/JPGs there or set `flag` fields in `config.AGENT_PRESETS`.

## Simulation knobs and rules (high level)
- Resources: `food`, `iron`, `oil`, `timber`, `rare_earths`; production is stability‑scaled and puppets tribute ~50% of new production to overlords.
- Research: economic (boosts tax), industrial (random production bumps), military (combat strength), social (population growth; 2‑turn cooldown).
- Infrastructure: expensive but permanent production boosts; rare earth exploration is 50% success.
- Market: targeted trades clear first at seller ask, then public order book with price–time priority; partial fills allowed, unfilled expire.
- War: HP = army×5, 20 rounds max, defend reduces next hit, victory grants gold/resource steal and +10% production to the winner.
- Bankruptcy/aid window: two turns at zero gold triggers insolvency; next turn opens bids (`aid_bid`) and forces the least‑bad offer.

## Resuming a run
Runs are resumable: rerun `python main.py --out runs/demo --turns 20 ...` and the engine auto‑loads the latest `turn_XXX/world.json`, restores memories, and continues from `turn+1`.

## Troubleshooting
- If a model returns malformed JSON, the engine retries once and logs parser notes in news plus `*_regen_diff.patch`.
- Missing assets/flags are silently skipped; verify filenames in `config.FLAGS_DIR_CANDIDATES`.
- To reduce VRAM pressure, the engine calls `/api/stop` between agents; override Ollama URL/timeout in `ollama_client.py` if your setup differs.
