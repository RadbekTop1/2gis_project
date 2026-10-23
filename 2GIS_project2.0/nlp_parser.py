"""
Интеллектуальный парсер пользовательских запросов.
Сочетает лемматизированный прямой поиск и TF-IDF сходство.
"""
import re
import pymorphy3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import ASPECT_PROFILES, DISH_PROFILES

morph = pymorphy3.MorphAnalyzer()

def lemmatize_text(text: str) -> str:
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", text.lower())
    lemmas = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(lemmas)

class SmartQueryParser:
    def __init__(self, sim_threshold: float = 0.08):
        self.sim_threshold = sim_threshold
        
        self.aspect_names = list(ASPECT_PROFILES.keys())
        # Храним множества лемм для моментального поиска точных совпадений
        self.aspect_lemma_sets = {k: set(lemmatize_text(ASPECT_PROFILES[k]).split()) for k in self.aspect_names}
        self.aspect_docs = [" ".join(self.aspect_lemma_sets[k]) for k in self.aspect_names]
        
        self.dish_names = list(DISH_PROFILES.keys())
        self.dish_lemma_sets = {k: set(lemmatize_text(DISH_PROFILES[k]).split()) for k in self.dish_names}
        self.dish_docs = [" ".join(self.dish_lemma_sets[k]) for k in self.dish_names]
        
        self.vec_aspects = TfidfVectorizer(ngram_range=(1, 2)).fit(self.aspect_docs)
        self.vec_dishes = TfidfVectorizer(ngram_range=(1, 2)).fit(self.dish_docs)
        
        self.matrix_aspects = self.vec_aspects.transform(self.aspect_docs)
        self.matrix_dishes = self.vec_dishes.transform(self.dish_docs)

    def parse(self, query: str):
        lemma_query = lemmatize_text(query)
        if not lemma_query.strip():
            return [], []

        query_tokens = set(lemma_query.split())

        # 1. Поиск аспектов К1 (Прямое попадание леммы ИЛИ семантическое сходство)
        q_vec_a = self.vec_aspects.transform([lemma_query])
        sims_a = cosine_similarity(q_vec_a, self.matrix_aspects)[0]
        
        matched_aspects = []
        for i, name in enumerate(self.aspect_names):
            # Если есть хотя бы одно точное словарное совпадение леммы
            has_direct_hit = bool(query_tokens & self.aspect_lemma_sets[name])
            if has_direct_hit or sims_a[i] >= self.sim_threshold:
                matched_aspects.append(name)

        # 2. Поиск блюд и сценариев
        q_vec_d = self.vec_dishes.transform([lemma_query])
        sims_d = cosine_similarity(q_vec_d, self.matrix_dishes)[0]
        
        matched_dishes = []
        for i, name in enumerate(self.dish_names):
            has_direct_hit = bool(query_tokens & self.dish_lemma_sets[name])
            if has_direct_hit or sims_d[i] >= self.sim_threshold:
                matched_dishes.append(name)

        return matched_aspects, matched_dishes

query_parser = SmartQueryParser()