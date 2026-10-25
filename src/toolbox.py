"""Tools exposed to the LLM + code-level guardrails (verification limits, ticket honesty)."""
import csv, json
from .config import DATASET
from .dates import date_calc
from tools import lookup_order, create_ticket  # dataset's mock tools

_S = lambda **p: {"type": "object", "properties": p}
SPECS = [
    {"name": "search_kb", "description": "Search ZentroMart policy/help/FAQ docs. English query with key terms.",
     "parameters": {**_S(query={"type": "string"}), "required": ["query"]}},
    {"name": "lookup_order", "description": "Order lookup. Call with order_id, then again with verification_value (phone last 4 digits OR delivery pincode). Details only when verified.",
     "parameters": {**_S(order_id={"type": "string"}, verification_value={"type": ["string", "null"]}), "required": ["order_id"]}},
    {"name": "create_ticket", "description": "Human handover. Summary: issue, intent, order ID, what customer said, what you checked, escalation reason. No phone/pincode.",
     "parameters": {**_S(summary={"type": "string"}, order_id={"type": ["string", "null"]}, reason={"type": ["string", "null"]}, intent={"type": ["string", "null"]}), "required": ["summary"]}},
    {"name": "lookup_product", "description": "Catalog lookup by SKU or name words (price, return_category, warranty_months, assured...).",
     "parameters": {**_S(query={"type": "string"}), "required": ["query"]}},
    {"name": "check_pincode", "description": "Serviceability/COD/Express for a pincode.",
     "parameters": {**_S(pincode={"type": "string"}), "required": ["pincode"]}},
    {"name": "date_calc", "description": "Exact date maths. ops: business_days_between(start,end), calendar_days_between, add_business_days(start,n), add_calendar_days(start,n). ISO dates; start defaults to today.",
     "parameters": {**_S(op={"type": "string", "enum": ["business_days_between", "calendar_days_between", "add_business_days", "add_calendar_days"]},
                         start={"type": ["string", "null"]}, end={"type": ["string", "null"]}, n={"type": ["integer", "null"]}), "required": ["op"]}},
]


def _csv(name):
    with open(DATASET / "data" / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


_PRODUCTS, _PINS = _csv("products.csv"), {r["pincode"]: r for r in _csv("pincodes.csv")}


class Toolbox:
    def __init__(self, kb):
        self.kb, self.order, self.fails = kb, None, {}
        self.calls, self.retrieved, self.tickets = [], [], []  # per-conversation logs

    def run(self, name, args):
        try:
            out = getattr(self, "t_" + name)(**args)
        except Exception as e:
            out = {"error": f"{type(e).__name__}: {e}"}
        self.calls.append({"name": name, "args": args, "ok": not (isinstance(out, dict) and "error" in out)})
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)

    META = ("KB-00.1", "KB-00.3")  # precedence + business-day rules (also embedded in the system prompt)

    def _add_meta(self):
        have = {x["id"] for x in self.retrieved}
        self.retrieved += [self.kb.by_id[i] for i in self.META if i in self.kb.by_id and i not in have]

    def register(self, hits):
        have = {x["id"] for x in self.retrieved}
        self.retrieved += [h for h in hits if h["id"] not in have]

    def t_search_kb(self, query):
        self._add_meta()
        hits = self.kb.search(query)
        self.register(hits)
        return self.kb.fmt(hits)

    def t_lookup_order(self, order_id, verification_value=None):
        oid = "".join(c for c in str(order_id) if c.isdigit())
        if self.fails.get(oid, 0) >= 2:
            return {"blocked": True, "message": "Verification failed twice. Do NOT share order details; offer general policy help and suggest contacting support from the registered phone number."}
        r = lookup_order(oid, verification_value)
        if verification_value and not r.get("verified"):
            self.fails[oid] = self.fails.get(oid, 0) + 1
            if self.fails[oid] == 1:
                r["message"] = "Verification failed. Allow ONE retry: ask again for the last 4 phone digits or delivery pincode."
        if r.get("verified"):
            self.order = r
        return r

    def t_create_ticket(self, summary, order_id=None, reason=None, intent=None):
        r = create_ticket(summary, order_id=order_id or (self.order or {}).get("order_id"), reason=reason, intent=intent)
        if r.get("created"):
            self.tickets.append(r["ticket_id"])
        return r

    def t_lookup_product(self, query):
        q = query.strip().lower()
        rows = [p for p in _PRODUCTS if p["sku"].lower() == q] or \
               [p for p in _PRODUCTS if all(w in (p.get("title", "") + " " + p["sku"]).lower() for w in q.split())]
        return [{k: v for k, v in p.items() if v} for p in rows[:5]] or {"found": False}

    def t_check_pincode(self, pincode):
        r = _PINS.get("".join(c for c in str(pincode) if c.isdigit()))
        return dict(r) if r else {"found": False, "message": "Pincode not in the serviceability table."}

    def t_date_calc(self, **kw):
        self._add_meta()
        return date_calc(**kw)
