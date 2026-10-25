"""Section-level knowledge-base index: TF-IDF (word + char n-gram) hybrid, status/precedence aware."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import DATASET

_HDR = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_SEC = re.compile(r"^## \[([A-Za-z0-9.\-]+)\]\s*(.*)$", re.M)
_TOK = re.compile(r"[\w\u0900-\u097F]+")
PRECEDENCE = {"policy": 0.06, "agent_guide": 0.045, "help": 0.03, "faq": 0.015, "marketing": 0.0, "meta": 0.0}


def _tok(t):
    return _TOK.findall(t.lower())


def _meta(block):
    return {k.strip(): v.strip() for k, v in (l.split(":", 1) for l in block.splitlines() if ":" in l)}


class KB:
    def __init__(self, kb_dir=None):
        self.chunks = []
        for f in sorted((kb_dir or DATASET / "kb").glob("*.md")):
            self._load(f)
        self.by_id = {c["id"]: c for c in self.chunks}
        texts = [c["index_text"] for c in self.chunks]
        self.vw = TfidfVectorizer(tokenizer=_tok, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.mw, self.mc = self.vw.fit_transform(texts), self.vc.fit_transform(texts)

    def _load(self, path):
        raw = path.read_text(encoding="utf-8")
        m = _HDR.match(raw)
        meta, body = (_meta(m.group(1)), raw[m.end():]) if m else ({}, raw)
        secs = list(_SEC.finditer(body))
        if not secs:
            secs_data = [(meta.get("doc_id", path.stem), meta.get("title", ""), body.strip())]
        else:
            secs_data = [(s.group(1), s.group(2), body[s.start():(secs[i + 1].start() if i + 1 < len(secs) else len(body))].strip())
                         for i, s in enumerate(secs)]
        for sid, title, text in secs_data:
            self.chunks.append({
                "id": sid, "section_title": title, "text": text, "file": path.name,
                "doc_title": meta.get("title", ""), "doc_type": meta.get("doc_type", ""),
                "status": meta.get("status", "current"), "effective_from": meta.get("effective_from", ""),
                "index_text": f"{meta.get('title', '')} {title} {text}",
            })

    def search(self, query, k=6):
        qw, qc = self.vw.transform([query]), self.vc.transform([query])
        sim = 0.6 * (self.mw @ qw.T).toarray().ravel() + 0.4 * (self.mc @ qc.T).toarray().ravel()
        for i, c in enumerate(self.chunks):
            sim[i] *= 0.6 if c["status"] != "current" else 1.0
            sim[i] += PRECEDENCE.get(c["doc_type"], 0) if sim[i] > 0 else 0
        top = np.argsort(-sim)[:k]
        return [{**{x: self.chunks[i][x] for x in ("id", "section_title", "text", "file", "doc_title", "doc_type", "status", "effective_from")},
                 "score": round(float(sim[i]), 4)} for i in top if sim[i] > 0.02]

    @staticmethod
    def fmt(hits):
        if not hits:
            return "No relevant document sections found."
        return "\n\n".join(f"[{h['id']}] {h['doc_title']} | {h['doc_type']} | status={h['status']} | effective_from={h['effective_from']}\n{h['text']}" for h in hits)
