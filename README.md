# VoiceOps 2026 – ZentroMart voice support agent

**audio → Gnani STT → agent (LLM + RAG + tools) → Gnani TTS → spoken reply**
GitHub:(https://github.com/anjali78p/voiceops_genai)

## 1. Install
```bash
pip install -r requirements.txt
cp .env.example .env     # Windows: copy .env.example .env
```
Fill `.env` (never commit it): `LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`, (`LLM_BASE_URL` for OpenAI-compatible providers such as Gemini/Groq/OpenRouter), `GNANI_API_KEY`, and `DATASET_DIR` = the extracted capstone dataset folder (the one containing `kb/`, `data/`, `tools/`). The dataset is not included in this repo.
STT/TTS use only Gnani (`gnani-vachana`). The LLM can be any hosted model with tool calling.

## 2. Run (batch – what the organizers use)
```bash
python run.py <folder_with_audio> --out-dir outputs [--stt-lang en-IN]
```
(`--audio-dir <folder>` also works.) For every audio file it saves `<name>.reply.txt`, `<name>.reply.wav`, `<name>.sources.json` (document chunks used) and `results.json` (heard text, reply, sources, tool calls). A failure on one file never stops the batch; every file still gets a spoken reply.
Offline text test without Gnani: put transcripts in `*.txt` files and add `--text-mode`.

## 3. Evaluate on the practice set
```bash
python -m src.evaluate baseline      # Step 1: prompt-only, no documents
python -m src.evaluate rag           # Step 2: RAG + tools
python -m src.evaluate convs         # Step 3: multi-turn conversations
python -m src.evaluate rag --ids PQ0008,PQ0022   # rerun only some items
python make_report.py --team "Name"  # builds report.pdf from eval_results/summary.json
```
Runs are cached and resumable (re-run the same command after a quota stop; `--fresh` restarts).

## 4. Design
- `src/kb.py` – section-level chunks keeping `[SECTION-ID]`, status, effective_from; word+char TF-IDF hybrid; superseded docs down-weighted; doc-type precedence.
- `src/agent.py` – tool-calling loop, per-turn pre-retrieval, verified-order state, empty-reply retry.
- `src/toolbox.py` – `search_kb`, `lookup_order`, `create_ticket`, `lookup_product`, `check_pincode`, `date_calc`; guardrails (one verification retry, ticket confirmed only when `created=True`).
- `src/dates.py` – deterministic business-day maths (Mon–Sat minus holidays, today = 2026-10-20).
- `src/voice.py` – Gnani STT/TTS, speech-friendly text normalisation, recovery on empty transcript.
- `src/evaluate.py` – Step 1/2/3 evaluation with caching.

## 5. Notes
No answers are hardcoded; all replies come from the agent. Only synthetic data is used. STT clips are limited to 60 s by the Gnani REST API.
