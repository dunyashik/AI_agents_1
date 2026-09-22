import time
import pandas as pd
from IPython.display import display

from helpers import load_tasks, ask_structured, is_correct, summary, compare_chart, ledger, MODELS
from agent import run_agent, load_cached_trace, save_trace
from tools import TOOLS_LIST


def run_evaluation(dataset_path: str):
    tasks = load_tasks(dataset_path)
    results = []

    configs = [
        {"name": "strong model, no tools", "model": MODELS["strong"], "tools": None},
        {"name": "cheap model, no tools", "model": MODELS["cheap"], "tools": None},
        {"name": "cheap model, wikipedia only", "model": MODELS["cheap"], "tools": [TOOLS_LIST[0]]},
        {"name": "cheap model, search two sources", "model": MODELS["cheap"], "tools": [TOOLS_LIST[0], TOOLS_LIST[2]]},
        {"name": "cheap model, wikipedia and page_find", "model": MODELS["cheap"],
         "tools": [TOOLS_LIST[0], TOOLS_LIST[1]]},
        {"name": "mid model, your best set", "model": MODELS["mid"], "tools": TOOLS_LIST},
    ]

    for conf in configs:
        correct = 0
        before_cost = ledger.total
        total_steps = 0
        started = time.perf_counter()

        for i, task in enumerate(tasks, 1):
            task_id = f"{conf['name']}_{task.get('id', 'task')}".replace(" ", "_").replace(",", "")

            if conf["tools"] is None:
                cached = load_cached_trace(task_id)
                if cached is not None:
                    ans, steps = cached
                else:
                    try:
                        ans = ask_structured(task["question"], conf["model"]).final
                    except Exception:
                        ans = "error"
                    steps = 1
                    save_trace(task_id, {"task_id": task_id, "question": task["question"]}, ans, steps)
            else:
                ans, steps = run_agent(task["question"], conf["model"], conf["tools"], task_id)

            total_steps += steps

            if is_correct(task, ans):
                correct += 1

        total_time = time.perf_counter() - started
        cost = ledger.total - before_cost
        n_tasks = len(tasks)

        res = summary(
            conf["name"],
            conf["model"],
            n_tasks,
            correct,
            cost,
            steps=total_steps / n_tasks if n_tasks else 1,
            seconds=total_time / n_tasks if n_tasks else 0.0
        )
        results.append(res)
        print(f"Configuration '{conf['name']}' completed. Accuracy: {res['accuracy']}")

    df_report = pd.DataFrame(results)

    print("\n--- FINAL REPORT ---")
    try:
        display(df_report)
    except NameError:
        print(df_report.to_markdown(index=False))

    fig = compare_chart(df_report)
    fig.savefig("money_chart.png")