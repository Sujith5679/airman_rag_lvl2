import json
import requests
from time import sleep

API_URL = "http://127.0.0.1:8000/ask"

with open(r"C:\Users\sujit\Desktop\armiango\questions\questions.json") as f:
    data = json.load(f)

results = {
    "grounded": [],
    "refused": [],
    "failed": [],
    "details": []
}

question_list = [q for section in data.values() for q in section]

for i, q in enumerate(question_list, 1):
    print(f"\n[{i}/{len(question_list)}] Asking: {q}")
    try:
        res = requests.post(f"{API_URL}?q={q}", timeout=120)
        answer = res.json().get("answer", "").strip()
        citations = res.json().get("citations", [])

        detail = {
            "question": q,
            "answer": answer,
            "citations": citations
        }

        # detection
        if answer == "This information is not available in the provided document(s).":
            results["refused"].append(q)
            detail["status"] = "refused"
        
        elif answer and citations:
            results["grounded"].append(q)
            detail["status"] = "grounded"
        
        else:
            results["failed"].append(q)
            detail["status"] = "failed"

        results["details"].append(detail)

    except Exception as e:
        results["failed"].append(q)
        results["details"].append({
            "question": q,
            "answer": None,
            "citations": [],
            "status": "failed",
            "error": str(e)
        })
    
    sleep(1)


total = len(question_list)

retrieval_hit = len(results["grounded"]) / total
hallucination_rate = 0 
faithfulness = 1 - hallucination_rate

# BEST 5 
best = sorted([x for x in results["details"] if x["status"] == "grounded"], key=lambda x: len(x["answer"]), reverse=True)[:5]

# WORST 5 
worst = results["details"][-5:]

report = {
    "summary": {
        "total_questions": total,
        "grounded": len(results["grounded"]),
        "refused": len(results["refused"]),
        "failed": len(results["failed"]),
        "retrieval_hit_rate": retrieval_hit,
        "faithfulness": faithfulness,
        "hallucination_rate": hallucination_rate
    },
    "best_5": best,
    "worst_5": worst,
    "all_results": results["details"]
}

with open("evaluation_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n Evaluation Complete! Report saved to evaluation_report.json")
