from evaluation import run_evaluation

if __name__ == "__main__":
    print("Starting evaluation...")
    run_evaluation("qa_data/fresh_2026.jsonl")
    print("Evaluation completed! Check the 'traces' folder and 'money_chart.png'.")