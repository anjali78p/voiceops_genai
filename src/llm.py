"""Provider-agnostic chat+tool-calling wrapper. Neutral message format:
 {"role":"user","content":str} | {"role":"assistant","content":str,"tool_calls":[{"id","name","args"}]}
 | {"role":"tool","tool_call_id","name","content":str}"""
import os, json, time


class LLM:
    def __init__(self, provider=None, model=None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
        self.model = model or os.getenv("LLM_MODEL") or {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}.get(self.provider, "")
        self.temp = float(os.getenv("LLM_TEMPERATURE")) if os.getenv("LLM_TEMPERATURE") else None
        self.usage = {"calls": 0, "in": 0, "out": 0}
        self._c = None

    def client(self):
        if self._c is None:
            key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY")
            if self.provider == "anthropic":
                import anthropic; self._c = anthropic.Anthropic(api_key=key)
            else:
                import openai; self._c = openai.OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL") or None)
        return self._c

    def chat(self, system, messages, tools=None, max_tokens=2500):
        for attempt in range(5):
            try:
                out = self._fake(messages) if self.provider == "fake" else (
                    self._anthropic(system, messages, tools, max_tokens) if self.provider == "anthropic"
                    else self._openai(system, messages, tools))
                self.usage["calls"] += 1
                return out
            except Exception as e:
                s = str(e).lower()
                hard = any(x in s for x in ("402", "credits", "per day", "perday", "invalid api key", "401"))
                retry = (not hard) and any(x in s for x in ("429", "rate", "tool_use_failed", "overload", "529", "503", "502", "500", "timeout", "connection"))
                if attempt == 4 or not retry:
                    raise
                time.sleep(5 * 2 ** attempt)

    def _anthropic(self, system, messages, tools, max_tokens):
        msgs = []
        for m in messages:
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = ([{"type": "text", "text": m["content"]}] if m.get("content") else []) + \
                         [{"type": "tool_use", "id": t["id"], "name": t["name"], "input": t["args"]} for t in m.get("tool_calls", [])]
                msgs.append({"role": "assistant", "content": blocks})
            else:
                blk = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                last = msgs[-1] if msgs else None
                if last and last["role"] == "user" and isinstance(last["content"], list):
                    last["content"].append(blk)
                else:
                    msgs.append({"role": "user", "content": [blk]})
        kw = dict(model=self.model, system=system, messages=msgs, max_tokens=max_tokens)
        if self.temp is not None:
            kw["temperature"] = self.temp
        if tools:
            kw["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        r = self.client().messages.create(**kw)
        self.usage["in"] += r.usage.input_tokens; self.usage["out"] += r.usage.output_tokens
        return {"text": "".join(b.text for b in r.content if b.type == "text"),
                "tool_calls": [{"id": b.id, "name": b.name, "args": b.input} for b in r.content if b.type == "tool_use"]}

    def _openai(self, system, messages, tools):
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                d = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    d["tool_calls"] = [{"id": t["id"], "type": "function", "function": {"name": t["name"], "arguments": json.dumps(t["args"])}} for t in m["tool_calls"]]
                msgs.append(d)
            elif m["role"] == "tool":
                msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                msgs.append({"role": "user", "content": m["content"]})
        kw = dict(model=self.model, messages=msgs, max_tokens=int(os.getenv("LLM_MAX_TOKENS", 2500)))
        if self.temp is not None:
            kw["temperature"] = self.temp
        if tools:
            kw["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in tools]
        if os.getenv("LLM_REASONING_EFFORT"):
            kw["reasoning_effort"] = os.getenv("LLM_REASONING_EFFORT")
        r = self.client().chat.completions.create(**kw)
        if r.usage:
            self.usage["in"] += r.usage.prompt_tokens; self.usage["out"] += r.usage.completion_tokens
        m = r.choices[0].message
        return {"text": m.content or "",
                "tool_calls": [{"id": t.id, "name": t.function.name, "args": json.loads(t.function.arguments or "{}")} for t in (m.tool_calls or [])]}

    def _fake(self, messages):  # offline plumbing test only
        if messages[-1]["role"] == "user":
            return {"text": "", "tool_calls": [{"id": "f1", "name": "search_kb", "args": {"query": messages[-1]["content"]}}]}
        return {"text": "Fake reply from offline test.", "tool_calls": []}
