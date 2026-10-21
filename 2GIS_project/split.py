import csv
import re

with open("data/reviews.csv", encoding="utf-8-sig") as f:
    r = list(csv.DictReader(f))

res = []
for i in r:
    for s in re.split(r"(?<=[.!?])\s+", i["text"]):
        if len(s) > 10:
            res.append([i["place"], i["review_id"], s, "", ""])

with open("data/sentences_all.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["place", "review_id", "text", "aspect", "sentiment"])
    w.writerows(res)

print(f"{len(res)} предложений")