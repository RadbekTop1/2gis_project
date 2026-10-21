"""
К2: профили заведений.
Читает predicted_all.csv (из k1.py predict) и reviews.csv, пишет:
    place_profiles.csv  - по каждой паре «заведение + аспект»
    place_dishes.csv    - по каждой паре «заведение + блюдо»
"""
import csv
import re
from math import sqrt
from collections import defaultdict

from config import ASPECT_KEYWORDS, DISHES

PREDICTED = "data/predicted_all.csv"
REVIEWS = "data/reviews.csv"
MIN_PROBA = 0.6     # предсказания К1 с меньшей уверенностью отбрасываем


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def wilson(pos, n, z=1.645):
    """Нижняя граница доверительного интервала доли положительных (90%).
    100% из 3 отзывов даёт ~0.52, 100% из 25 даёт ~0.89: чем больше данных, тем выше оценка."""
    if n == 0:
        return 0.0
    p = pos / n
    centre = p + z * z / (2 * n)
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


def build_aspects():
    # Одно упоминание = один отзыв (несколько предложений об одном аспекте в отзыве считаются 1 раз)
    per_review = defaultdict(list)
    for r in read_csv(PREDICTED):
        if float(r["aspect_proba"]) >= MIN_PROBA and r["aspect"] in ASPECT_KEYWORDS:
            per_review[(r["place"], r["review_id"], r["aspect"])].append(int(r["sentiment"]))

    counts = defaultdict(lambda: [0, 0])  # (place, aspect) -> [pos, neg]
    for (place, _, aspect), sents in per_review.items():
        total = sum(sents)
        if total > 0:
            counts[(place, aspect)][0] += 1
        elif total < 0:
            counts[(place, aspect)][1] += 1

    rows = []
    for (place, aspect), (p, n) in sorted(counts.items()):
        total = p + n
        rows.append([place, aspect, total, p, n, round(p / total, 3), round(wilson(p, total), 3)])
    return rows


def build_dishes():
    compiled = {name: re.compile(rx) for name, rx in DISHES.items()}
    counts = defaultdict(lambda: [0, 0])  # (place, dish) -> [отзывов с оценкой 4-5, всего]
    for r in read_csv(REVIEWS):
        text = r["text"].lower()
        good = int(r["rating"]) >= 4
        for name, rx in compiled.items():
            if rx.search(text):
                counts[(r["place"], name)][0] += good
                counts[(r["place"], name)][1] += 1

    rows = []
    for (place, dish), (g, t) in sorted(counts.items()):
        rows.append([place, dish, t, g, round(g / t, 3), round(wilson(g, t), 3)])
    return rows


def write(path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    aspects = build_aspects()
    write("data/place_profiles.csv", ["place", "aspect", "mentions", "pos", "neg", "share_pos", "score"], aspects)
    dishes = build_dishes()
    write("data/place_dishes.csv", ["place", "dish", "mentions", "good", "share_good", "score"], dishes)
    print(f"data/place_profiles.csv: {len(aspects)} строк, заведений: {len({r[0] for r in aspects})}")
    print(f"data/place_dishes.csv: {len(dishes)} строк, заведений: {len({r[0] for r in dishes})}")
