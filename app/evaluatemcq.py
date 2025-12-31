import json
import requests
from time import sleep
from statistics import mean

API_URL = "http://127.0.0.1:8000/ask"
JSON_FILE = r"C:\Users\sujit\Desktop\armiango\questions\mcqquestions.json"
OUTPUT_FILE = "evaluation_mcq_lvl2_report.json"


def detect_choice(answer, options):
    """Match MCQ option by text or letter; handles INFERRED responses"""
    ans = answer.lower().replace("inferred:", "").strip()

    for key, text in options.items():
        if text.lower() in ans:
            return key

    for key in options.keys():
        if key.lower() in ans:
            return key

    return None


def evaluate_mcq():
    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    results = []
    correct = refused = failed = grounded = 0
    retrieval_hits = []

    print("\n Level 2 MCQ Evaluation Started...\n")

    for q in data["questions"]:
        q_id = q["id"]
        question = q["question"]
        options = q["options"]

        print(f"\n Question {q_id}: {question}")

        try:
            res = requests.post(API_URL, params={"q": question, "debug": "true"}, timeout=200)
            res_data = res.json()
            answer = res_data.get("answer", "").strip()
            citations = res_data.get("citations", [])
            retrieval_hits.append(1 if citations else 0)

        except Exception as e:
            failed += 1
            results.append({
                "id": q_id,
                "question": question,
                "model_answer": None,
                "status": " failed (timeout/error)",
                "error": str(e)
            })
            continue

        matched_choice = detect_choice(answer, options)

        if "not available" in answer.lower():
            refused += 1
            status = " refused (correct behavior)"

        elif matched_choice is not None:
            correct += 1
            status = f" correct (matched {matched_choice})"
            if citations:
                grounded += 1

        else:
            failed += 1
            status = " hallucination (answer doesn't match any option)"

        results.append({
            "id": q_id,
            "question": question,
            "model_answer": answer,
            "matched_choice": matched_choice,
            "citations": citations,
            "status": status
        })

        sleep(1)

    total = len(data["questions"])
    accuracy = round(correct / total, 2)
    retrieval_rate = round(mean(retrieval_hits), 2)
    faithfulness = round(grounded / total, 2)
    hallucination_rate = round(failed / total, 2)

    report = {
        "evaluation_type": "MCQ – Level 2 (Hybrid Retrieval)",
        "metrics": {
            "total_questions": total,
            "correct": correct,
            "refused": refused,
            "failed": failed,
            "choice_matching_accuracy": accuracy,
            "retrieval_hit_rate": retrieval_rate,
            "faithfulness_rate": faithfulness,
            "hallucination_rate": hallucination_rate
        },
        "question_breakdown": results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print("\n MCQ Evaluation Complete! Report saved as:", OUTPUT_FILE)
    print("\n SUMMARY:")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    evaluate_mcq()
