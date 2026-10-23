import json
import csv

with open("dataset.json", encoding="utf-8") as file:
    places = json.load(file)

place_rows = []
for place in places:
    rubrics_text = ", ".join(place["rubrics"])
    place_rows.append([place["title"], rubrics_text, place["address"], place["totalScore"], place["url"]])

with open("data/places.csv", "w", encoding="utf-8-sig", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["place", "rubrics", "address", "total_score", "url"])
    writer.writerows(place_rows)

print(f"{len(place_rows)} заведений")