"""
К3: запрос пользователя -> рекомендации.
    python k3.py "хочу с детьми, кофе и бургеры"
    python k3.py            (спросит запрос в консоли)
"""
import csv
from itertools import combinations
import os
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

    requested_items_count = len(aspects) + len(dishes)

    results = []
    for place in places:
        found_scores = []
        notes = []
        matched_features = []

        for aspect in aspects:
            profile = profiles.get((place, aspect))
            if profile:
                mentions = int(profile["mentions"])
                found_scores.append(float(profile["score"]))
                matched_features.append(aspect)
                notes.append(f"{aspect}: {mentions} упоминаний, {float(profile['share_pos']):.0%} позитива (скор: {float(profile['score']):.2f})")
            else:
                notes.append(f"{aspect}: данных нет")

        for dish in dishes:
            dish_profile = dish_data.get((place, dish))
            if dish_profile:
                mentions = int(dish_profile["mentions"])
                found_scores.append(float(dish_profile["score"]))
                matched_features.append(dish)
                notes.append(f"{dish}: в {mentions} отзывах, {float(dish_profile['share_good']):.0%} 4-5★ (скор: {float(dish_profile['score']):.2f})")
            else:
                notes.append(f"{dish}: данных нет")

        if not matched_features:
            continue

        coverage = len(found_scores) / requested_items_count
        base_score = sum(found_scores) / len(found_scores) * coverage

        combo_bonus = 0.0
        if len(matched_features) >= 2:
            for combo in combinations(sorted(matched_features), 2):
                combo_name = " + ".join(combo)
                if (place, combo_name) in combos_data:
                    combo_info = combos_data[(place, combo_name)]
                    combo_bonus += 0.15
                    notes.append(f"★ Найдена связка К2: [{combo_name}] (подтверждений: {combo_info['confirmations']})")

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

    results.sort(key=lambda result: result["score"], reverse=True)
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