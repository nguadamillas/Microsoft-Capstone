"""
Procurement Assistant — FastAPI backend.

Exposes POST /chat. Runs an Anthropic tool-use loop where the model computes exact
figures from the Gold tables via a sandboxed `query_data` tool, then returns a
structured answer the frontend can render as text + KPI cards + a chart.
"""
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from data import load_gold, run_query, schema_context

load_dotenv()

MODEL = os.environ.get("ASSISTANT_MODEL", "claude-sonnet-4-6")  # confirm your team's string
MAX_TOOL_HOPS = 6

TABLES, USING_SAMPLE = load_gold()
SCHEMA = schema_context(TABLES)

# Startup log — visible in uvicorn terminal, confirms real vs. sample data.
print(f"[assistant] using_sample_data={USING_SAMPLE}", file=sys.stderr)
for _n, _df in TABLES.items():
    print(f"[assistant]   {_n}: {len(_df):,} rows", file=sys.stderr)

SYSTEM_PROMPT = f"""You are the Procurement Assistant for "TED Procurement Intelligence"
(IE University x Microsoft), a business product for European public-procurement buyers and suppliers.

Scope and grounding:
- Answer ONLY questions about European public procurement, using ONLY the project's Gold tables below.
- Use the query_data tool to compute every figure. Never invent numbers, table names, or columns.
- If a question is outside procurement, or the data cannot answer it, say so briefly and suggest a few
  questions you CAN answer. Do not attempt unrelated topics.

Data integrity — CRITICAL:
- The "(example rows — column FORMAT ONLY)" lines in each TABLE block are 2 illustrative rows showing
  column types and formats. They are NOT the dataset and MUST NOT be used as figures.
- Every TABLE header shows the real row count (e.g. "8,000 rows total"). Your answers must reflect ALL
  those rows, not just the 2 examples.
- You MUST call query_data for EVERY number, count, ranking, sum, or percentage you report. No exceptions.
- If a query_data result starts with "ERROR", fix the expression using the exact column names from the
  schema and call query_data again. Do not answer from the example rows under any circumstances.

Voice: a concise procurement strategist. Explain what a figure means for buyer/supplier strategy, not
how the model works. Keep answers tight.

Finishing: once you have the figures, reply with ONLY a single JSON object (no prose, no code fences):
{{
  "text": "<short markdown business answer>",
  "kpis": [{{"label": "<short label>", "value": "<formatted value e.g. EUR 4.2B / 18.9% / 440>"}}],
  "chart": {{"type": "bar|line|doughnut", "title": "<title>", "labels": [...],
             "series": [{{"name": "<name>", "data": [...]}}]}},
  "source": "<gold table(s) used>"
}}
Include "kpis" and/or "chart" ONLY when the answer is quantitative or comparative; omit them otherwise.
Always include "text" and "source". Format large euro amounts compactly (K/M/B).

GOLD TABLE SCHEMAS:
{SCHEMA}
"""

TOOLS = [{
    "name": "query_data",
    "description": "Evaluate a single pandas expression over the Gold DataFrames "
                   "(available by name, plus `pd`) and return the result as text.",
    "input_schema": {
        "type": "object",
        "properties": {"pandas_expression": {
            "type": "string",
            "description": "A pandas expression, e.g. "
                           "gold_awards.groupby('buyer_country').size().sort_values(ascending=False).head(5)",
        }},
        "required": ["pandas_expression"],
    },
}]

app = FastAPI(title="Procurement Assistant")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list[dict]  # [{role: "user"|"assistant", content: "..."}]


def _client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to assistant/backend/.env")
    return anthropic.Anthropic(api_key=key)


def _coerce_answer(raw_text: str) -> dict:
    """Parse the model's final JSON answer; fall back to plain text if needed."""
    raw = raw_text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "text" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return {"text": raw_text, "source": ""}


@app.get("/health")
def health():
    return {"ok": True, "using_sample_data": USING_SAMPLE, "model": MODEL}


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        client = _client()
    except RuntimeError as exc:
        return {"error": str(exc)}

    messages = [{"role": m["role"], "content": m["content"]} for m in req.messages]

    try:
        for hop in range(MAX_TOOL_HOPS):
            kwargs = dict(
                model=MODEL, max_tokens=1200, system=SYSTEM_PROMPT,
                tools=TOOLS, messages=messages,
            )
            if hop == 0:
                # Force the model to call query_data at least once before answering.
                kwargs["tool_choice"] = {"type": "any"}
            resp = client.messages.create(**kwargs)

            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                tool_results = []
                for block in resp.content:
                    if block.type == "tool_use" and block.name == "query_data":
                        expr = block.input.get("pandas_expression", "")
                        out = run_query(expr, TABLES)
                        print(
                            f"[assistant] query_data: {expr}\n"
                            f"             -> {out[:200]}",
                            file=sys.stderr,
                        )
                        tool_results.append({
                            "type": "tool_result", "tool_use_id": block.id, "content": out,
                        })
                messages.append({"role": "user", "content": tool_results})
                continue

            # Final turn — gather text blocks and parse the JSON answer.
            text = "".join(b.text for b in resp.content if b.type == "text")
            answer = _coerce_answer(text)
            answer["using_sample_data"] = USING_SAMPLE
            return answer

        return {"error": "The assistant took too many steps. Try a more specific question."}
    except Exception as exc:  # network / API / parsing
        return {"error": f"{type(exc).__name__}: {exc}"}
