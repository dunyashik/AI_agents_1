import json
import re
from pathlib import Path

from helpers import chat
from tools import TOOL_FUNCTIONS

Path("traces").mkdir(exist_ok=True)

AGENT_SYSTEM = """You are a helpful assistant answering questions about recent events (up to 2026).
You MUST use tools to verify facts. Do not answer from memory.
Use web_search or ddg_search first to find the context.
If reading a broad Wikipedia article (like '2026 in science'), use page_find with the EXACT title and specific keywords to find the exact detail.
If the question needs arithmetic, use calculator. If it needs something easier to compute than to reason about (e.g. counting days between dates), use python_exec.
Do not repeat the exact same tool call twice.
When you are ready to answer, your final message MUST end with exactly: "FINAL: <short answer>". Do not use JSON for the final answer."""


def extract_final(content: str, fallback: str = None) -> str:
    content = content or ""
    m = re.search(r"FINAL:\s*(.+)", content)
    if m:
        return m.group(1).strip()
    return content.strip() if fallback is None else fallback


def save_trace(task_id: str, trace: dict, final_answer: str, step_count: int) -> None:
    trace["final_answer"] = final_answer
    trace["steps_taken"] = step_count
    with open(f"traces/{task_id}.json", "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)


def call_tool(call: dict, seen_calls: set) -> str:
    func_name = call["function"]["name"]
    args_str = call["function"]["arguments"]
    call_signature = f"{func_name}({args_str})"

    if call_signature in seen_calls:
        return "Error: You already made this exact tool call. Try different keywords or a different tool."

    seen_calls.add(call_signature)
    try:
        args_dict = json.loads(args_str)
        return str(TOOL_FUNCTIONS[func_name](**args_dict))[:2500]
    except Exception as e:
        return f"Error executing {func_name}: {str(e)}"


def load_cached_trace(task_id: str):
    path = Path(f"traces/{task_id}.json")
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["final_answer"], cached["steps_taken"]
    except (json.JSONDecodeError, KeyError):
        return None


def run_agent(question: str, model: str, tools_list: list, task_id: str, max_steps: int = 8):
    cached = load_cached_trace(task_id)
    if cached is not None:
        return cached

    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": question}
    ]
    seen_calls = set()
    trace = {"task_id": task_id, "question": question, "steps": []}
    step_count = 0

    for _ in range(max_steps):
        step_count += 1
        try:
            msg = chat(messages, model, tools=tools_list, tag=f"agent_{task_id}", temperature=0)
        except RuntimeError:
            save_trace(task_id, trace, "unknown", step_count)
            return "unknown", step_count

        messages.append(msg)
        trace["steps"].append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg.get("tool_calls")})

        calls = msg.get("tool_calls")
        if not calls:
            final_ans = extract_final(msg.get("content"))
            save_trace(task_id, trace, final_ans, step_count)
            return final_ans, step_count

        for c in calls:
            result = call_tool(c, seen_calls)
            tool_msg = {"role": "tool", "tool_call_id": c["id"], "content": result}
            messages.append(tool_msg)
            trace["steps"].append(tool_msg)

    step_count += 1
    messages.append({"role": "user", "content": "Limit reached. Give your best answer starting with FINAL:"})
    msg = chat(messages, model, tag=f"agent_{task_id}_force_finish", temperature=0)
    final_ans = extract_final(msg.get("content"), fallback="unknown")

    save_trace(task_id, trace, final_ans, step_count)
    return final_ans, step_count