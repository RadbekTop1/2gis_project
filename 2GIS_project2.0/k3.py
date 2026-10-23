"""
К3: запрос пользователя -> рекомендации.
    python k3.py "хочу с детьми, кофе и бургеры"
    python k3.py            (спросит запрос в консоли)
"""
import csv
from itertools import combinations
import os
import re
import sys

from nlp_parser import query_parser

TOP = 5


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_query(query):
    return query_parser.parse(query)


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

    combos_data = (
        {(r["place"], r["combo"]): r for r in read_csv("data/place_combos.csv")}
        if os.path.exists("data/place_combos.csv")
        else {}
    )

    results = []
    for place in places:
        parts, notes = [], []
        matched_features = []

        # 1. Проверяем аспекты
        for a in aspects:
            r = profiles.get((place, a))
            if r:
                n = int(r["mentions"])
                parts.append(float(r["score"]))
                matched_features.append(a)
                notes.append(f"{a}: {n} упоминаний, {float(r['share_pos']):.0%} позитива (скор: {float(r['score']):.2f})")
            else:
                parts.append(0.35)
                notes.append(f"{a}: данных нет (нейтрально)")

        # 2. Проверяем блюда
        for d in dishes:
            r = dish_data.get((place, d))
            if r:
                n = int(r["mentions"])
                parts.append(float(r["score"]))
                matched_features.append(d)
                notes.append(f"{d}: в {n} отзывах, {float(r['share_good']):.0%} 4-5★ (скор: {float(r['score']):.2f})")
            else:
                parts.append(0.35)
                notes.append(f"{d}: данных нет (нейтрально)")

        # Если вообще ни одного совпадения по запросу нет — пропускаем
        if not matched_features:
            continue

        base_score = sum(parts) / len(parts)

        # 3. Бонус за подтвержденные связки К2 (синергия запроса)
        combo_bonus = 0.0
        if len(matched_features) >= 2:
            for c in combinations(sorted(matched_features), 2):
                c_name = " + ".join(c)
                if (place, c_name) in combos_data:
                    c_info = combos_data[(place, c_name)]
                    combo_bonus += 0.15
                    notes.append(f"★ Найдена связка К2: [{c_name}] (подтверждений: {c_info['confirmations']})")

        extra = info.get(place, {})
        rating_bonus = (float(extra["total_score"]) / 5) if extra.get("total_score") else 0.5

        final_score = 0.75 * base_score + 0.15 * min(combo_bonus, 0.3) + 0.10 * rating_bonus

        results.append({
            "place": place, 
            "score": round(final_score, 3), 
            "notes": notes,
            "rubrics": extra.get("rubrics", ""), 
            "address": extra.get("address", "")
        })

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