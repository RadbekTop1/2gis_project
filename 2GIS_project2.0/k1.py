"""
К1: аспекты + тональность по предложениям отзывов.

Запуск:
    python k1.py train      # обучить, оценить, сохранить модели в k1_models.joblib
    python k1.py predict    # разметить sentences_all.csv -> predicted_all.csv
"""
import csv
import sys
import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

LABELED = "data/labeled_all.csv"
REVIEWS = "data/reviews.csv"
SENTENCES = "data/sentences_all.csv"
MODELS = "k1_models.joblib"
OUTPUT = "data/predicted_all.csv"
NO_ASPECT = "нет"
SEED = 1


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_ratings():
    return {r["review_id"]: int(r["rating"]) for r in read_csv(REVIEWS)}


def rating_features(review_ids, ratings):
    """One-hot по оценке отзыва 1..5 (неизвестная оценка -> нули)."""
    m = np.zeros((len(review_ids), 5))
    for i, rid in enumerate(review_ids):
        r = ratings.get(rid)
        if r:
            m[i, r - 1] = 1
    return csr_matrix(m)


def make_vectorizer():
    # char_wb (2..5): ловит словоформы и опечатки, работает и для казахских слов
    return TfidfVectorizer(ngram_range=(2, 5), analyzer="char_wb", min_df=2, sublinear_tf=True)


def load_labeled(ratings):
    """Ключ примера = (review_id, text): одинаковые фразы из разных отзывов не склеиваются."""
    by_key = {}
    for row in read_csv(LABELED):
        key = (row["review_id"], row["text"])
        by_key.setdefault(key, [])
        if row["aspect"] != NO_ASPECT:  # «нет аспекта» = пустой набор, а не отдельный класс
            by_key[key].append((row["aspect"], int(row["sentiment"])))
    keys = list(by_key)
    return keys, [by_key[k] for k in keys]


def fit_aspect_model(texts, Y):
    vec = make_vectorizer()
    X = vec.fit_transform(texts)
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X, Y)
    return vec, model


def tune_threshold(texts, Y, groups):
    """Подбор порога по out-of-fold вероятностям на train (группировка по отзыву)."""
    vec = make_vectorizer()
    X = vec.fit_transform(texts)
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    proba = cross_val_predict(model, X, Y, groups=groups, cv=GroupKFold(5), method="predict_proba")
    best_t, best_f = 0.5, -1
    for t in np.arange(0.2, 0.71, 0.05):
        f = f1_score(Y, proba >= t, average="micro", zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    print(f"порог аспектов: {best_t:.2f} (out-of-fold micro-F1 {best_f:.2f})")
    return best_t


def sentiment_inputs(items, ratings):
    """items: список (review_id, text, aspect) -> (тексты, матрица признаков рейтинга)."""
    texts = [f"{a} | {t}" for _, t, a in items]
    return texts, rating_features([rid for rid, _, _ in items], ratings)


def fit_sentiment_model(items, y, ratings):
    texts, rf = sentiment_inputs(items, ratings)
    vec = make_vectorizer()
    X = hstack([vec.fit_transform(texts), rf]).tocsr()
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return vec, model


def predict_sentiment(vec, model, items, ratings):
    texts, rf = sentiment_inputs(items, ratings)
    return model.predict(hstack([vec.transform(texts), rf]).tocsr())


def train():
    ratings = load_ratings()
    keys, labels = load_labeled(ratings)
    texts = [t for _, t in keys]
    rids = [r for r, _ in keys]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform([[a for a, _ in lab] for lab in labels])
    print(f"примеров: {len(keys)}, отзывов: {len(set(rids))}, аспектов: {len(mlb.classes_)}")

    # Деление по отзыву: предложения одного отзыва не попадают и в train, и в test
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(splitter.split(texts, groups=rids))
    tr_texts = [texts[i] for i in tr]
    te_texts = [texts[i] for i in te]

    # --- Аспекты ---
    threshold = tune_threshold(tr_texts, Y[tr], [rids[i] for i in tr])
    a_vec, a_model = fit_aspect_model(tr_texts, Y[tr])
    proba = a_model.predict_proba(a_vec.transform(te_texts))
    pred = proba >= threshold
    print("\n=== АСПЕКТЫ (test, деление по отзыву) ===")
    print(classification_report(Y[te], pred, target_names=mlb.classes_, zero_division=0))

    # --- Тональность ---
    def pairs(idx):
        items, y = [], []
        for i in idx:
            for a, s in labels[i]:
                items.append((rids[i], texts[i], a))
                y.append(s)
        return items, y

    tr_items, tr_y = pairs(tr)
    te_items, te_y = pairs(te)
    s_vec, s_model = fit_sentiment_model(tr_items, tr_y, ratings)
    pred_s = predict_sentiment(s_vec, s_model, te_items, ratings)
    print("=== ТОНАЛЬНОСТЬ (на истинных аспектах) ===")
    print(classification_report(te_y, pred_s, zero_division=0))

    # --- Сквозная проверка: предсказанные аспекты -> предсказанная тональность ---
    true_pairs, pred_pairs = set(), set()
    pred_items = []
    for row, i in enumerate(te):
        for a, s in labels[i]:
            true_pairs.add((i, a, s))
        for j in np.where(pred[row])[0]:
            pred_items.append((i, mlb.classes_[j]))
    if pred_items:
        items = [(rids[i], texts[i], a) for i, a in pred_items]
        sents = predict_sentiment(s_vec, s_model, items, ratings)
        pred_pairs = {(i, a, int(s)) for (i, a), s in zip(pred_items, sents)}
    hit = len(true_pairs & pred_pairs)
    p = hit / len(pred_pairs) if pred_pairs else 0
    r = hit / len(true_pairs) if true_pairs else 0
    f = 2 * p * r / (p + r) if p + r else 0
    print(f"=== СКВОЗНАЯ ОЦЕНКА (аспект+тональность) === P={p:.2f} R={r:.2f} F1={f:.2f}")

    # --- Финальные модели на всех данных ---
    a_vec, a_model = fit_aspect_model(texts, Y)
    all_items, all_y = pairs(range(len(keys)))
    s_vec, s_model = fit_sentiment_model(all_items, all_y, ratings)
    joblib.dump(
        {"mlb": mlb, "a_vec": a_vec, "a_model": a_model, "threshold": threshold,
         "s_vec": s_vec, "s_model": s_model},
        MODELS,
    )
    print(f"\nмодели сохранены: {MODELS}")


def predict():
    ratings = load_ratings()
    bundle = joblib.load(MODELS)
    rows = read_csv(SENTENCES)
    texts = [r["text"] for r in rows]
    proba = bundle["a_model"].predict_proba(bundle["a_vec"].transform(texts))
    classes = bundle["mlb"].classes_

    hits = []  # (индекс строки, аспект, вероятность)
    for i in range(len(rows)):
        for j in np.where(proba[i] >= bundle["threshold"])[0]:
            hits.append((i, classes[j], float(proba[i, j])))

    items = [(rows[i]["review_id"], rows[i]["text"], a) for i, a, _ in hits]
    sents = predict_sentiment(bundle["s_vec"], bundle["s_model"], items, ratings) if items else []

    with open(OUTPUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["place", "review_id", "text", "aspect", "sentiment", "aspect_proba"])
        for (i, a, p), s in zip(hits, sents):
            r = rows[i]
            w.writerow([r["place"], r["review_id"], r["text"], a, int(s), f"{p:.3f}"])
    print(f"предложений: {len(rows)}, с аспектом: {len({h[0] for h in hits})}, пар аспект-тональность: {len(hits)}")
    print(f"результат: {OUTPUT}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": train, "predict": predict}[mode]()