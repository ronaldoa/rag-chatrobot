import csv, json
inp="data/eval.csv"; out="data/eval.jsonl"
with open(inp, newline='', encoding='utf-8') as f, open(out, "w", encoding='utf-8') as g:
    for row in csv.DictReader(f):
        g.write(json.dumps({"question": row.get("question",""), "answer": row.get("answer","")})+"\n")
print("written", out)
