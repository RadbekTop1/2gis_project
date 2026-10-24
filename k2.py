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
from nlp_parser import lemmatize_text

from config import ASPECT_PROFILES, DISH_PROFILES
# Логика К2 для связок на совместных упоминаниях в одном отзыве
from itertools import combinations

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

def make_dish_lemma_sets():
    dish_lemma_sets = {}
    for dish_name in DISH_PROFILES:
        dish_lemma_sets[dish_name] = set(lemmatize_text(DISH_PROFILES[dish_name]).split())
    return dish_lemma_sets

def build_aspects():
    # Одно упоминание = один отзыв (несколько предложений об одном аспекте в отзыве считаются 1 раз)
    per_review = defaultdict(list)
    for r in read_csv(PREDICTED):
        if float(r["aspect_proba"]) >= MIN_PROBA and r["aspect"] in ASPECT_PROFILES:
            per_review[(r["place"], r["review_id"], r["aspect"])].append(int(r["sentiment"]))

    counts = defaultdict(lambda: [0, 0])  # (place, aspect) -> [pos, neg]
    for (place, _, aspect), sents in per_review.items():
        # Считаем количество положительных (1) и отрицательных (-1) предложений
        pos_sents = sum(1 for s in sents if s == 1)
        neg_sents = sum(1 for s in sents if s == -1)
        
        # Если преобладает позитив — засчитываем как +1 к позитивным отзывам
        if pos_sents > neg_sents:
            counts[(place, aspect)][0] += 1
        # Если преобладает негатив — засчитываем в негативные
        elif neg_sents > pos_sents:
            counts[(place, aspect)][1] += 1

    rows = []
    for (place, aspect), (p, n) in sorted(counts.items()):
        total = p + n
        rows.append([place, aspect, total, p, n, round(p / total, 3), round(wilson(p, total), 3)])
    return rows
def build_combos(min_support=2):
    """
    Формирует связки характеристик на основе совместных позитивных упоминаний
    в рамках одного и того же отзыва.
    """
    # Собираем все позитивные аспекты и блюда для каждого отзыва
    review_features = defaultdict(set)
    
    # 1. Аспекты из предсказаний К1
    for r in read_csv(PREDICTED):
        if float(r["aspect_proba"]) >= MIN_PROBA and int(r["sentiment"]) == 1:
            if r["aspect"] in ASPECT_PROFILES:
                review_features[(r["place"], r["review_id"])].add(r["aspect"])
                
    # 2. Блюда из текста того же отзыва
    dish_lemma_sets = make_dish_lemma_sets()
    for review in read_csv(REVIEWS):
        if int(review["rating"]) >= 4:
            review_lemmas = set(lemmatize_text(review["text"]).split())
            for dish_name in dish_lemma_sets:
                if review_lemmas & dish_lemma_sets[dish_name]:
                    review_features[(review["place"], review["review_id"])].add(dish_name)

    # Подсчитываем частоту совместного появления пар характеристик
    combo_counts = defaultdict(lambda: defaultdict(int))
    for (place, _), feats in review_features.items():
        if len(feats) >= 2:
            for combo in combinations(sorted(feats), 2):
                combo_key = " + ".join(combo)
                combo_counts[place][combo_key] += 1

    rows = []
    for place, combos in combo_counts.items():
        for combo_name, count in combos.items():
            if count >= min_support:
                # Уровень подтверждения по количеству совместных отзывов
                level = "Высокий" if count >= 5 else "Средний"
                rows.append([place, combo_name, count, level])
                
    return rows

def build_dishes():
    dish_lemma_sets = make_dish_lemma_sets()
    counts = defaultdict(lambda: [0, 0])
    for review in read_csv(REVIEWS):
        review_lemmas = set(lemmatize_text(review["text"]).split())
        is_good = int(review["rating"]) >= 4
        for dish_name in dish_lemma_sets:
            if review_lemmas & dish_lemma_sets[dish_name]:
                counts[(review["place"], dish_name)][0] += is_good
                counts[(review["place"], dish_name)][1] += 1

    rows = []
    for (place, dish), (good_count, total_count) in sorted(counts.items()):
        rows.append([place, dish, total_count, good_count, round(good_count / total_count, 3), round(wilson(good_count, total_count), 3)])
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
    combos = build_combos()
    write("data/place_combos.csv", ["place", "combo", "confirmations", "level"], combos)
    print(f"data/place_combos.csv: {len(combos)} связок сохранено")
    write("data/place_dishes.csv", ["place", "dish", "mentions", "good", "share_good", "score"], dishes)
    print(f"data/place_profiles.csv: {len(aspects)} строк, заведений: {len({r[0] for r in aspects})}")
    print(f"data/place_dishes.csv: {len(dishes)} строк, заведений: {len({r[0] for r in dishes})}")
