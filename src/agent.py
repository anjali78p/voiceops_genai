import json
from .config import TODAY
from .kb import KB
from .llm import LLM
from .toolbox import Toolbox, SPECS

_KB = None
def shared_kb():
    global _KB
    if _KB is None:
        _KB = KB()
    return _KB

BASELINE = ("You are ZentroMart's AI customer support assistant (an Indian online store for groceries and electronics). "
            "In your first reply tell the customer you are an AI assistant. Answer the customer's question as best you can. Keep replies to 2-3 short sentences.")

RULES = """You are ZentroMart's AI customer support assistant (Indian online store: groceries & electronics). Today: Tuesday, 20 October 2026.
- OPENING: your first reply says you are ZentroMart's AI support assistant.
- GROUNDING: answer ONLY from tool results and the pre-retrieved sections. Use search_kb (English queries) for policy/process/timeline questions; lookup_product for SKUs (warranty: product warranty_months AND any Assured extension in the policy); check_pincode for delivery/COD/Express; date_calc for ALL date arithmetic. If the documents don't cover it, say you don't have that information and offer a human colleague. Never invent policies, dates or amounts.
- PRECEDENCE: current policy > agent guide > help > FAQ > marketing; ignore superseded sections. Check special rules (sale/festival period by the order's placed date, Assured, category).
- BUSINESS DAYS: Mon-Sat except 2026-08-15, 2026-10-02, 2026-10-19, 2026-11-09 (refund/logistics timelines). Return windows are calendar days.
- ANSWER FIRST: answer general policy/how-to questions right away (e.g. what to do for a damaged item); ask for the order ID only when order-specific facts are needed.
- ORDER DATA: share nothing about a specific order before verification: ask the order ID, call lookup_order(order_id), then ask ONE of last 4 phone digits / delivery pincode and call lookup_order again with it. One retry only. Never read out phone/email/address. Use only the VERIFIED ORDER block below. When a verification detail arrives and passes, confirm in one short sentence and give the status relevant to their earlier problem; do NOT create a ticket in that turn.
- DEADLINES: return deadline = delivery date + window days (date_calc add_calendar_days); also mention the reverse-pickup fee if the policy charges it to this customer (check Plus status). Refund timing ("by when / how long"): take the MAX of the REF-2 range for the payment method, use date_calc add_business_days from the initiation date, and answer 'by 21 October 2026'. When discussing a refund, quote its reference number (RRN) from the order record.
- ACTIONS: never claim something is done unless a tool succeeded (ticket: created=true). You cannot cancel, refund or approve anything.
- HANDOVER: call create_ticket FIRST, then say a colleague will contact them; promise no outcome/amount/time; never say 'approved'. Triggers: damaged/wrong item over Rs 10,000 (check price with lookup_product); order/account activity they did not make; legal/consumer-court/police; safety risk; customer asks for a human; customer says this is their 3rd+ contact or contacts_last_7_days >= 2; refund more than 3 business days past its max timeline. If the customer's own message shows a trigger, create the ticket in that same turn without waiting for an order ID (write 'order not verified yet' in the summary). Politely refuse policy-breaking requests (e.g. extra compensation) and hand over if the policy says a human decides. Create tickets only for what the customer is asking now, not for facts in the order record they have not raised.
- CONVERSATION: remember what they said; resolve follow-ups from context; ask ONE short question for a missing detail. Input may be speech-to-text (digits spelled out, no punctuation): interpret sensibly and ask to repeat unclear IDs.
- STYLE: 2-3 short sentences; reply in the customer's language (English / Hindi / Hinglish in Latin script as they write). Write dates in full as '27 October 2026' (in Hindi use the Hindi month name); write counts, durations and amounts as digits (3 attempts, 48 hours, 12 months, Rs 500)."""

VOICE = """
VOICE CHANNEL: Replies are spoken aloud. No markdown, bullets, emojis or symbols. Say order IDs in groups of four digits, amounts in words with 'rupees', dates like 'the 24th of October 2026'. Max 2-3 short sentences."""


class Agent:
    def __init__(self, mode="rag", voice=False, llm=None, kb=None, max_steps=6):
        self.mode, self.voice, self.llm, self.max_steps = mode, voice, llm or LLM(), max_steps
        self.tb = Toolbox(kb or shared_kb()) if mode == "rag" else None
        self.history, self.trace = [], []

    def _system(self):
        if self.mode != "rag":
            return BASELINE
        s = RULES + (VOICE if self.voice else "")
        if getattr(self, "_ctx", ""):
            s += "\n\nPRE-RETRIEVED DOCUMENT SECTIONS (may be incomplete; call search_kb for more):\n" + self._ctx
        o = self.tb.order
        if o:
            keep = {k: v for k, v in o.items() if v not in (None, "") and k not in ("found", "verified", "customer_id")}
            s += "\n\nVERIFIED ORDER (this conversation):\n" + json.dumps(keep, ensure_ascii=False, default=str)
        return s

    def respond(self, text):
        msgs = self.history + [{"role": "user", "content": text}]
        n_calls, n_tix = (len(self.tb.calls), len(self.tb.tickets)) if self.tb else (0, 0)
        reply, empty_retry = None, False
        if self.tb:
            q = " ".join([m["content"] for m in self.history if m["role"] == "user"][-1:] + [text])
            hits = self.tb.kb.search(q, k=3)
            self.tb.register(hits); self.tb._add_meta()
            self._ctx = self.tb.kb.fmt(hits)
        for _ in range(self.max_steps):
            out = self.llm.chat(self._system(), msgs, SPECS if self.tb else None)
            if not out["tool_calls"] or not self.tb:
                reply = out["text"].strip()
                if reply or empty_retry:
                    break
                empty_retry = True
                msgs.append({"role": "user", "content": "(Please give your final reply to the customer now.)"})
                continue
            msgs.append({"role": "assistant", "content": out["text"], "tool_calls": out["tool_calls"]})
            for tc in out["tool_calls"]:
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": self.tb.run(tc["name"], tc["args"])})
        reply = reply or "Sorry, I'm having trouble right now. Let me connect you with a human colleague."
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        self.trace.append({"user": text, "reply": reply,
                           "tool_calls": self.tb.calls[n_calls:] if self.tb else [],
                           "ticket_created": bool(self.tb and len(self.tb.tickets) > n_tix)})
        return reply

    @property
    def retrieved_ids(self):
        return [c["id"] for c in self.tb.retrieved] if self.tb else []
