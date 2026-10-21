"""
К3: запрос пользователя -> рекомендации.
    python k3.py "хочу с детьми, кофе и бургеры"
    python k3.py            (спросит запрос в консоли)
"""
import csv
import os
import re
import sys


from config import ASPECT_KEYWORDS, DISHES

TOP = 5


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_query(query):
    q = query.lower()
    aspects = [a for a, rx in ASPECT_KEYWORDS.items() if re.search(rx, q)]
    dishes = [d for d, rx in DISHES.items() if re.search(rx, q)]
    return aspects, dishes


def load_data():
    profiles = {(r["place"], r["aspect"]): r for r in read_csv("data/place_profiles.csv")}
    dishes = {(r["place"], r["dish"]): r for r in read_csv("data/place_dishes.csv")}
    info = {r["place"]: r for r in read_csv("data/places.csv")} if os.path.exists("data/places.csv") else {}
    places = {p for p, _ in profiles} | {p for p, _ in dishes}
    return profiles, dishes, info, places


def recommend(query, top=TOP):
    aspects, dishes = parse_query(query)
    if not aspects and not dishes:
        return aspects, dishes, []
    profiles, dish_data, info, places = load_data()

    results = []
    for place in places:
        parts, notes = [], []
        for a in aspects:
            r = profiles.get((place, a))
            if r:
                n = int(r["mentions"])
                parts.append(float(r["score"]))
                notes.append(f"{a}: {n} упоминаний, {float(r['share_pos']):.0%} положительных")
            else:
                parts.append(0)
                notes.append(f"{a}: данных нет")
        for d in dishes:
            r = dish_data.get((place, d))
            if r:
                n = int(r["mentions"])
                parts.append(float(r["score"]))
                notes.append(f"{d}: в {n} отзывах, {float(r['share_good']):.0%} с оценкой 4-5★")
            else:
                parts.append(0)
                notes.append(f"{d}: данных нет")
        if not any(p > 0 for p in parts):
            continue
        score = sum(parts) / len(parts)
        rating_bonus = 0.0
        extra = info.get(place, {})
        if extra.get("total_score"):
            rating_bonus = float(extra["total_score"]) / 5   # общий рейтинг 2GIS, 0..1
        score = 0.95 * score + 0.05 * rating_bonus
        results.append({"place": place, "score": score, "notes": notes,
                        "rubrics": extra.get("rubrics", ""), "address": extra.get("address", "")})
    results.sort(key=lambda r: r["score"], reverse=True)
    return aspects, dishes, results[:top]


def main():
    query = " ".join(sys.argv[1:]) or input("Куда хотите пойти? ")
    aspects, dishes, results = recommend(query)
    if not aspects and not dishes:
        print("Не нашёл в запросе знакомых тем. Попробуйте слова вроде: кофе, десерты, дети, недорого, быстро, "
              "уютно, бургер, пицца, плов.")
        return
    print(f"Понял запрос: аспекты = {aspects or '-'}, блюда = {dishes or '-'}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['place']}  (счёт {r['score']:.2f})")
        if r["rubrics"] or r["address"]:
            print(f"   {r['rubrics']} | {r['address']}")
        for n in r["notes"]:
            print(f"   - {n}")
        print()


if __name__ == "__main__":
    main()