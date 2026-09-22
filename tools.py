import ast
import operator
import re
import subprocess
import sys
import time
import requests
from pydantic import BaseModel, Field

WIKI = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "agents-course-homework/1.0"}

def wiki(params):
    for attempt in range(3):
        try:
            r = requests.get(WIKI, params={**params, "format": "json"}, headers=WIKI_HEADERS, timeout=20)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                return r.json()["query"]
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("Wikipedia is unreachable")

class WebSearchArgs(BaseModel):
    query: str = Field(description="short query: a name, a title, a term")

def web_search(query: str) -> str:
    try:
        hits = wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": 3}).get("search", [])
        if not hits: return "nothing found"
        pages = wiki({"action": "query", "prop": "extracts", "explaintext": 1, "exintro": 1, "titles": hits[0]["title"]}).get("pages", {})
        text = " ".join(next(iter(pages.values())).get("extract", "").split())[:1500]
        others = ", ".join(h["title"] for h in hits[1:])
        return f"[{hits[0]['title']}] {text}" + (f" | other articles: {others}" if others else "")
    except Exception as e:
        return f"web_search error: {e}"

class PageFindArgs(BaseModel):
    title: str = Field(description="exact article title")
    keywords: str = Field(description="two to five keywords from the question")

def page_find(title: str, keywords: str) -> str:
    try:
        pages = wiki({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "redirects": 1}).get("pages", {})
        text = next(iter(pages.values())).get("extract", "")
        if not text: return "no such page"
        words = [w for w in re.findall(r"\w+", keywords.lower()) if len(w) > 2]
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        hits = [l for l in lines if sum(w in l.lower() for w in words) >= max(1, (len(words) + 1) // 2)]
        return "\n".join(h[:400] for h in hits[:6]) or "keywords not found on the page"
    except Exception as e:
        return f"page_find error: {e}"

class DDGSearchArgs(BaseModel):
    query: str = Field(description="search query")

def ddg_search(query: str) -> str:
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=3))
        if not results: return "nothing found"
        return "\n".join(f"[{r.get('title', '')}] {r.get('body', '')}" for r in results)
    except Exception as e:
        return f"ddg_search error: {e}"

CALC_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}

def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in CALC_OPS:
        return CALC_OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in CALC_OPS:
        return CALC_OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("only numbers and arithmetic are allowed")

class CalculatorArgs(BaseModel):
    expr: str = Field(description="arithmetic expression, e.g. 17*23+5")

def calculator(expr: str) -> str:
    try:
        val = _evaluate(ast.parse(expr.replace(",", "."), mode="eval").body)
        return str(int(val)) if float(val).is_integer() else f"{val:.4f}".rstrip("0")
    except Exception as e:
        return f"calculator error: {e}"

class PythonExecArgs(BaseModel):
    code: str = Field(description="Python code to run; print() the result to see it")

def python_exec(code: str) -> str:
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return "code exceeded the 5 second limit"
    out = (r.stdout + r.stderr).strip()
    return out[:256] if out else "code ran but printed nothing"

TOOLS_LIST = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches English Wikipedia and returns the intro of the best article.",
            "parameters": WebSearchArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "page_find",
            "description": "Looks inside the full text of a Wikipedia article and returns lines containing keywords.",
            "parameters": PageFindArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ddg_search",
            "description": "Searches the web via DuckDuckGo and returns summaries of top results.",
            "parameters": DDGSearchArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates an arithmetic expression: numbers, parentheses, + - * / ** %.",
            "parameters": CalculatorArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Runs Python code in a separate process and returns what it printed. Use for date math, counting, or anything easier to compute than to reason about.",
            "parameters": PythonExecArgs.model_json_schema()
        }
    }
]

TOOL_FUNCTIONS = {
    "web_search": web_search,
    "page_find": page_find,
    "ddg_search": ddg_search,
    "calculator": calculator,
    "python_exec": python_exec
}