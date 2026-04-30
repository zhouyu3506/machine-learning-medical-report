from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.18

def predict_relevance(case):
    current_text = case.current_study.study_description or ""

    prior_texts = [
        study.study_description or ""
        for study in case.prior_studies
    ]

    if len(prior_texts) == 0:
        return []

    texts = [current_text] + prior_texts

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english"
    )

    vectors = vectorizer.fit_transform(texts)

    current_vec = vectors[0]
    prior_vecs = vectors[1:]

    sims = cosine_similarity(current_vec, prior_vecs)[0]

    results = []
    for study, sim in zip(case.prior_studies, sims):
        is_relevant = sim > THRESHOLD
        results.append((study.study_id, bool(is_relevant)))

    return results