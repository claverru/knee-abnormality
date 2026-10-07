import json

s = json.load(open("outputs/autoresearch/routing-and-shifts/state.json"))
for name in ("shared-ligament-reader", "height-cuts-from-the-spots", "contusion-reads-the-edema-head"):
    idea = s["ideas"][name]
    print("==", name, {k: v for k, v in idea.items() if k in ("status", "decision", "blocked_reason")})
    for key, exp in s["experiments"].items():
        if exp["idea"] == name:
            report = exp.get("developer_report") or {}
            review = exp.get("implementation_review") or {}
            print("  ", key[-24:], "stage", exp["stage"], "| report", report.get("status"), "|", str(report.get("blocked_reason") or report.get("rationale") or "")[:500])
            print("      review", review.get("decision"), [c[:300] for c in review.get("concerns", [])][:3])
