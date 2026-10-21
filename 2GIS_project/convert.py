import json
import csv

with open("dataset.json", encoding="utf-8") as f:
    d = json.load(f)

res = []
for p in d:
    for i in p["reviews"]:
        res.append([p["title"], i["id"], i["rating"], i["text"].replace("\n", " "), i["dateCreated"][:10]])

with open("data/reviews.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["place", "review_id", "rating", "text", "date"])
    w.writerows(res)

print(f"{len(res)} отзывов")


