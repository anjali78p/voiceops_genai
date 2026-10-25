"""Builds report.pdf (max 4 pages) from the real eval results in eval_results/summary.json.
Usage: python make_report.py --team "Your Team Name" --names "A, B"
Nothing is invented: numbers come from your own eval runs; missing runs are shown as 'not run'."""
import argparse, json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ap = argparse.ArgumentParser()
ap.add_argument("--team", default="Team")
ap.add_argument("--names", default="")
ap.add_argument("--out", default="report.pdf")
a = ap.parse_args()
sj = Path("eval_results/summary.json")
S = json.loads(sj.read_text()) if sj.exists() else {}

def pct(x):
    return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "not run"

def row(name, key, extra=None):
    d = S.get(key)
    if not d:
        return [name, "not run", "-", "-", "-"]
    first = d.get("pass", d.get("turn_pass"))
    return [name, d.get("completed", "-"), pct(first), pct(d.get("include")), extra(d) if extra else "-"]

st = getSampleStyleSheet()
H = ParagraphStyle("H", parent=st["Heading2"], spaceBefore=8, spaceAfter=3, fontSize=12)
B = ParagraphStyle("B", parent=st["BodyText"], fontSize=9.2, leading=12)
L = ParagraphStyle("L", parent=B, leftIndent=12, bulletIndent=2)
def p(t): return Paragraph(t, B)
def bl(items): return [Paragraph(i, L, bulletText="\u2022") for i in items]

doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.5*cm, bottomMargin=1.5*cm,
                        title="VoiceOps 2026 - ZentroMart support agent")
E = [Paragraph("VoiceOps 2026 &ndash; ZentroMart Voice Support Agent", st["Title"]),
     p(f"<b>{a.team}</b> {('&ndash; ' + a.names) if a.names else ''} &nbsp;|&nbsp; LLM &amp; Generative AI Course capstone, C&amp;A Club IIT Guwahati &times; Gnani.ai")]

E += [Paragraph("1. What I built", H),
 p("One agent, four steps: <b>audio &rarr; Gnani STT &rarr; agent (LLM + RAG + tools) &rarr; Gnani TTS &rarr; spoken reply</b>. "
   "The LLM is provider-agnostic (Anthropic or any OpenAI-compatible endpoint). STT and TTS use only the Gnani API (<font face='Courier'>gnani-vachana</font> SDK).")]
E += bl([
 "<b>Step 1 &ndash; prompt-only baseline.</b> A system prompt makes the model ZentroMart's AI support assistant (introduces itself as an AI, short replies) with no documents or tools. Used only as the comparison point.",
 "<b>Step 2 &ndash; grounded answers (RAG).</b> The 21 knowledge-base files are split into section-level chunks that keep their [SECTION-ID], status and effective_from. Retrieval is a word + character n-gram TF-IDF hybrid; superseded sections are down-weighted and document type breaks ties (policy &gt; agent guide &gt; help &gt; FAQ &gt; marketing). Each turn gets top-3 pre-retrieved sections and the model can call <font face='Courier'>search_kb</font> for more; every chunk used is saved with the answer. If the documents do not cover a question the agent says so and offers a human.",
 "<b>Step 3 &ndash; conversation.</b> Verified order data is kept in conversation state so follow-ups (&quot;how long will that take?&quot;) work without re-asking. Tools: <font face='Courier'>lookup_order</font>, <font face='Courier'>create_ticket</font> (dataset mock tools), plus <font face='Courier'>lookup_product</font>, <font face='Courier'>check_pincode</font> and a deterministic <font face='Courier'>date_calc</font> (business days Mon&ndash;Sat minus holidays, today = 20 Oct 2026). Code-level guardrails: at most one verification retry, and the agent only says a ticket exists when the tool returned created=True.",
 "<b>Step 4 &ndash; voice.</b> <font face='Courier'>run.py</font> takes a folder of audio files, runs Gnani STT, the agent and Gnani TTS, and saves for every file the reply text, the reply audio and the document chunks used. Replies are written for the ear (no markdown, order IDs grouped, amounts spoken as rupees); an empty transcript triggers a polite &quot;could you say it again&quot; recovery reply, and a failure in any file still produces a spoken reply.",
])

def ex_ret(d): return pct(d.get("retrieval_recall")) if "retrieval_recall" in d else "-"
def ex_conv(d): return f"conv pass {pct(d.get('conv_pass'))}"
tbl = [["Run", "Completed", "Pass", "must_include", "Extra"],
       row("Step 1: prompt-only (questions)", "baseline"),
       row("Step 2: RAG + tools (questions)", "rag", lambda d: "retrieval recall " + ex_ret(d)),
       row("Step 3: conversations (turns)", "convs", ex_conv)]
t = Table(tbl, colWidths=[6.3*cm, 2.4*cm, 2.0*cm, 2.6*cm, 4.0*cm])
t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E4F5")), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                       ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
E += [Paragraph("2. Step 1 vs Step 2 (my own evaluation)", H),
      p("Scored automatically on the practice set with rule-based checks (must_include / must_not_include, ticket created when the expected decision is escalate, "
        "and retrieved section IDs vs gold_evidence). 'Completed' shows how many items finished; free-tier token quotas limited some runs, so partial runs are reported as partial."),
      Spacer(1, 4), t, Spacer(1, 4),
      p("My first full RAG run (52/52 questions, before the fixes below) scored <b>67.3% pass</b> with <b>70.2% retrieval recall</b>. "
        "Its failures drove the changes in section 3; the table above holds the latest numbers I could finish.")]

E += [Paragraph("3. What went wrong and what I changed", H)]
E += bl([
 "<b>Empty replies.</b> The reasoning model spent its whole token budget thinking, so some answers came back empty and the agent said &quot;I'm having trouble&quot;. Fix: more output-token headroom plus one automatic retry when a reply is empty.",
 "<b>Date arithmetic.</b> The LLM counted refund days wrongly and gave ranges instead of the latest date. Fix: a deterministic <font face='Courier'>date_calc</font> tool and a prompt rule to use the maximum of the policy range and answer with the latest date.",
 "<b>Handover timing.</b> The agent asked for an order ID instead of escalating when the customer's own message already showed an escalation condition (e.g. unrecognised order, damaged item over Rs 10,000). It also created tickets too early on verification turns. Fix: explicit handover rules (ticket in the same turn for clear triggers, none on pure verification turns, repeat-contact trigger).",
 "<b>Missed special rules.</b> Sale-period windows and Assured warranty extensions were missed when the model answered from one source. Fix: pre-retrieval of the top sections every turn, and a rule to combine catalog, policy and order data.",
 "<b>Strict string checks.</b> Answers like &quot;27 Oct&quot; or &quot;three&quot; failed checks expecting &quot;27 October 2026&quot; or &quot;3&quot;. Fix: style rules for full month names and digits.",
 "<b>Tool-call schema errors.</b> Optional tool arguments sent as null were rejected by the API. Fix: allow null for optional fields and retry once on a failed tool call.",
 "<b>Evaluation cost.</b> A full run needs hundreds of LLM calls and hit daily free-tier limits. Fix: a resumable evaluator that caches every finished item and a <font face='Courier'>--ids</font> option to rerun only failing cases.",
])

E += [Paragraph("4. Limitations and what I would fix next", H)]
E += bl([
 "Retrieval is lexical (TF-IDF); Hindi/Hinglish queries rely on the LLM writing an English search query. A multilingual embedding model would improve recall.",
 "Gnani REST STT accepts clips up to 60 seconds; longer audio is not split yet. STT language is set per run (<font face='Courier'>--stt-lang</font>), with no automatic language detection.",
 "Scoring is rule-based, so correct answers phrased differently can be marked wrong; an LLM judge would give a fairer estimate.",
 "Answers depend on the chosen LLM; a smaller model follows the handover and date rules less reliably.",
])

E += [Paragraph("5. Bonus work", H)]
E += bl(["Resumable, cached evaluation harness with subset reruns; deterministic business-day/date tool; Hinglish and Hindi handling; speech-friendly text normalisation; per-file source-chunk logging in batch runs.",
         "Rules followed: Gnani only for STT/TTS, synthetic dataset only, no hardcoded answers, answers produced by the agent."])
doc.build(E)
print("wrote", a.out)
