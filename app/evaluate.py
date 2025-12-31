import json
import requests
from statistics import mean

API_URL = "http://127.0.0.1:8000/ask"
INPUT_FILE = r"C:\Users\sujit\Desktop\airmanc\questions\questions.json"
OUTPUT_FILE = "evaluation_report_lvl2.json"

def evaluate():
    data = json.load(open(INPUT_FILE))
    questions = []
    for category, q_list in data.items():
        for q in q_list:
            questions.append({"category": category, "question": q})
    
    results = []
    grounded = refused = failed = 0
    retrieval_hits = []

    print("\n Evaluating Open Questions...\n")

    for item in questions:
        query = item["question"]
        category = item["category"]

        try:
            response = requests.post(
                API_URL, params={"q": query, "debug": "true"}, timeout=120
            ).json()
        except Exception as e:
            print(f" ERROR (timeout or server issue): {query}")
            failed += 1
            continue

        answer = response.get("answer", "").lower()
        citations = response.get("citations", [])
        retrieval_hits.append(1 if citations else 0)
        if "not available" in answer:
            refused += 1
            status = "refused"
        elif citations:
            grounded += 1
            status = "grounded"
        else:
            failed += 1
            status = "failed"

        results.append({
            "category": category,
            "question": query,
            "answer": response.get("answer"),
            "citations": citations,
            "status": status
        })

    total = len(questions)
    report = {
        "evaluation_type": "open-ended",
        "total_questions": total,
        "grounded": grounded,
        "refused": refused,
        "failed": failed,
        "retrieval_hit_rate": round(mean(retrieval_hits), 2),
        "faithfulness": round(grounded / total, 2),
        "hallucination_rate": round(failed / total, 2),
        "sample_results": results[:5],  # preview
        "full_results": results
    }

    json.dump(report, open(OUTPUT_FILE, "w"), indent=4)
    print("\n Evaluation Complete!")
    print(f" Saved to {OUTPUT_FILE}")
    print("\n Summary:\n")
    print(json.dumps({
        "grounded": grounded,
        "refused": refused,
        "failed": failed,
        "retrieval_hit_rate": report["retrieval_hit_rate"],
        "faithfulness": report["faithfulness"],
        "hallucination_rate": report["hallucination_rate"]
    }, indent=2))


if __name__ == "__main__":
    evaluate()
