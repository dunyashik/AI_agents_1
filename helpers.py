import os
import re
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to a .env file in the project root.")

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://postypashki.ru",
    "X-Title": "agents-course-homework"
}

MODELS = {
    "cheap": "openai/gpt-4o-mini",
    "mid": "anthropic/claude-haiku-4.5",
    "strong": "anthropic/claude-sonnet-4.6"
}
COLORS = {"violet": "#5436A3", "amber": "#F09000", "teal": "#00838F", "red": "#C43C3C", "grey": "#787882"}

@dataclass
class Ledger:
    calls: list = field(default_factory=list)

    def add(self, tag, model, usage, seconds=0.0):
        p = usage.get("prompt_tokens", 0)
        c = usage.get("completion_tokens", 0)
        cost = usage.get("cost") or 0.0
        self.calls.append({"tag": tag, "model": model.split("/")[-1], "prompt": p, "completion": c,
                           "cost": cost, "seconds": round(seconds, 2)})
        return cost

    @property
    def total(self):
        return sum(c["cost"] for c in self.calls)

ledger = Ledger()

def post_with_retry(body, attempts=5):
    for attempt in range(attempts):
        r = requests.post(CHAT_URL, json=body, headers=HEADERS, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503) and attempt < attempts - 1:
            time.sleep(1.5 * (attempt + 1))
            continue
        raise RuntimeError(f"HTTP {r.status_code} : {r.text[:5000]}")

def chat(messages, model, tools=None, tag="chat", temperature=None):
    body = {"model": model, "messages": messages, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if temperature is not None:
        body["temperature"] = temperature
    started = time.perf_counter()
    data = post_with_retry(body)
    ledger.add(tag, model, data.get("usage") or {}, time.perf_counter() - started)
    return data["choices"][0]["message"]

class Answer(BaseModel):
    reasoning: str = Field(description="brief reasoning in two to three sentences")
    final: str = Field(description="short final answer only: number, name, title, or brief phrase")

SCHEMA_PROMPT = (
    "Answer strictly with one JSON object matching this schema, without any text around, no tabs or newlines:\n"
    + json.dumps(Answer.model_json_schema(), ensure_ascii=False)
)

def json_from(text):
    m = re.search(r"\[.*\]|\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON found in text")
    cleaned = re.sub(r'(\\["\\/bfnrtu])|\\', lambda e: e.group(1) or "\\\\", m.group(0))
    return json.loads(cleaned, strict=False)

def ask_structured(question, model, schema=Answer, attempts=3):
    messages = [
        {"role": "system", "content": SCHEMA_PROMPT},
        {"role": "user", "content": question},
    ]
    for _ in range(attempts):
        msg = chat(messages, model, tag="struct")
        try:
            return schema.model_validate(json_from(msg["content"]))
        except (ValidationError, ValueError) as e:
            messages += [
                msg,
                {"role": "user", "content": f"Answer failed schema validation: {e}. Return strict JSON matching schema."},
            ]
    raise ValueError("Model failed structured validation")

def load_tasks(name):
    return [json.loads(line) for line in Path(name).read_text(encoding="utf-8").splitlines() if line.strip()]

def normalize(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def is_correct(task, answer):
    if task.get("source") == "gsm8k":
        nums = re.findall(r"-?\d+(?:\.\d+)?", str(answer).replace(",", ""))
        return bool(nums) and abs(float(nums[-1]) - float(task["answer"])) < 1e-6
    return normalize(task["answer"]) in normalize(answer)

def summary(config, model, n, correct, cost, steps, seconds=0.0):
    return {
        "config": config,
        "model": model.split("/")[-1],
        "n": n,
        "accuracy": round(correct / n, 2) if n else 0,
        "cost_per_task": round(cost / n, 5) if n else 0,
        "cost_per_correct": round(cost / correct, 5) if correct else float("inf"),
        "avg_steps": round(steps, 2),
        "avg_seconds": round(seconds, 1),
    }

def compare_chart(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].bar(df["model"], df["accuracy"] * 100, color=COLORS["violet"])
    axes[0].set_ylabel("Accuracy, %")
    axes[0].set_ylim(0, 105)

    finite_costs = [c for c in df["cost_per_correct"] if c != float("inf")]
    max_finite = max(finite_costs) * 100 if finite_costs else 10
    display_costs = [c * 100 if c != float("inf") else max_finite * 1.2 for c in df["cost_per_correct"]]

    axes[1].bar(df["model"], display_costs, color=COLORS["amber"])
    axes[1].set_ylabel("Cost per correct, cents")

    axes[2].bar(df["model"], df["avg_seconds"], color=COLORS["teal"])
    axes[2].set_ylabel("Seconds per task")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    return fig