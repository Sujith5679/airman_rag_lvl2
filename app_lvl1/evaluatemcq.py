import json
import requests
from statistics import mean
from time import sleep

API_URL = "http://127.0.0.1:8000/ask"
INPUT_FILE = r"C:\Users\sujit\Desktop\airmanc\questions\mcqquestions.json"
OUTPUT_FILE = "evaluation_mcq_report_lvl2.json"


def detect_choice(answer, options):
    """Match the model output to a choice key A/B/C/D"""
    ans = answer.lower()

    # match by full option text
    for key, text in options.items():
        if text.lower() in ans:
            return key

    # match by letter mention
    for key in options.keys():
        if key.lower() in ans:
            return key

    return None


def evaluate_mcq():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    questions = data["questions"]

    results = []
    grounded = refused = failed = correct = 0
    retrieval_hits = []

    print("\n Evaluating MCQ Questions...\n")

    for q in questions:
        q_id = q["id"]
        query = q["question"].strip()
        options = q["options"]

        # ensure question format
        if not query.endswith("?"):
            query += "?"

        # API request
        try:
            res = requests.get(API_URL, params={"q": query, "debug": "true"}, timeout=120)
            response = res.json()
        except Exception as e:
            failed += 1
            results.append({
                "id": q_id,
                "question": query,
                "model_answer": None,
                "status": "failed (timeout/error)",
                "error": str(e)
            })
            continue

        answer = response.get("answer", "").strip()
        citations = response.get("citations", [])
        retrieval_hits.append(1 if citations else 0)

        matched_choice = detect_choice(answer, options)

        # CLASSIFY RESULT
        if answer == "This information is not available in the provided document(s).":
            refused += 1
            status = "refused"

        elif matched_choice is not None:
            correct += 1
            grounded += 1
            status = f"answered (matched choice: {matched_choice})"

        else:
            failed += 1
            status = "failed (no matching option)"

        results.append({
            "id": q_id,
            "question": query,
            "model_answer": answer,
            "matched_choice": matched_choice,
            "citations": citations,
            "status": status
        })

        sleep(1)  # avoid overloading API

    total = len(questions)
    accuracy = round(correct / total, 2)

    # 📌 Final Report ONLY Level 2 MCQ Results
    report = {
        "evaluation_type": "MCQ",
        "total_questions": total,
        "correct_choice_detected": correct,
        "refused": refused,
        "failed": failed,
        "retrieval_hit_rate": round(mean(retrieval_hits), 2),
        "choice_matching_accuracy_rate": accuracy,
        "faithfulness_rate": round(grounded / total, 2),
        "hallucination_rate": round(failed / total, 2),
        "sample_results": results[:5],
        "full_results": results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=4)

    print("\n Evaluation complete → Saved to:", OUTPUT_FILE)
    print("\n SUMMARY\n")
    print(json.dumps({
        "correct_choice_detected": correct,
        "refused": refused,
        "failed": failed,
        "accuracy": accuracy,
        "retrieval_hit_rate": round(mean(retrieval_hits), 2),
        "faithfulness": round(grounded / total, 2),
        "hallucination_rate": round(failed / total, 2)
    }, indent=2))


if __name__ == "__main__":
    evaluate_mcq()
    print("\n MCQ Evaluation Complete! ✨")
