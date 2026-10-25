"""Batch run: python run.py --audio-dir <folder> [--out-dir outputs]
For every audio file: Gnani STT -> agent (LLM + RAG + tools) -> Gnani TTS.
Saves per file: <stem>.reply.txt, <stem>.reply.wav, <stem>.sources.json; plus results.json."""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from src.agent import Agent, shared_kb
from src.llm import LLM
from src.voice import Voice

AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_pos", nargs="?", help="audio folder (same as --audio-dir)")
    ap.add_argument("--audio-dir")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--stt-lang", default="en-IN", help="Gnani STT language_code (e.g. en-IN, hi-IN, gu-IN)")
    ap.add_argument("--text-mode", action="store_true", help="treat *.txt files as transcripts; skip Gnani (offline test)")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if not (a.audio_dir or a.audio_pos):
        ap.error("give the audio folder: python run.py <folder>  (or --audio-dir <folder>)")
    src, out = Path(a.audio_dir or a.audio_pos), Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in ({".txt"} if a.text_mode else AUDIO))
    if not files:
        sys.exit(f"No input files found in {src}")
    shared_kb(); llm = LLM(); voice = None if a.text_mode else Voice()

    def one(p):
        rec = {"file": p.name, "heard": "", "reply": "", "sources": [], "tool_calls": [], "error": None}
        try:
            heard = p.read_text(encoding="utf-8").strip() if a.text_mode else voice.listen(p, a.stt_lang)
            rec["heard"] = heard
            if not heard:
                rec["reply"] = Voice.RECOVERY if voice else "(no input)"
            else:
                ag = Agent(mode="rag", voice=True, llm=llm)
                rec["reply"] = ag.respond(heard)
                rec["sources"] = [{k: c[k] for k in ("id", "file", "doc_title", "status", "text")} for c in ag.tb.retrieved]
                rec["tool_calls"] = ag.tb.calls
        except Exception as e:  # never leave a file without a spoken reply
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["reply"] = "Sorry, I ran into a problem. A human colleague will get back to you."
        (out / f"{p.stem}.reply.txt").write_text(rec["reply"], encoding="utf-8")
        (out / f"{p.stem}.sources.json").write_text(json.dumps(rec["sources"], ensure_ascii=False, indent=2), encoding="utf-8")
        if voice:
            rec["audio_ok"] = voice.speak(rec["reply"], out / f"{p.stem}.reply.wav")
        print(f"{p.name}: {rec['heard'][:60]!r} -> {rec['reply'][:70]!r}")
        return rec

    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, files))
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done: {len(results)} files -> {out}/  (LLM usage: {llm.usage})")


if __name__ == "__main__":
    main()
