import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.16

MODALITIES = {
    "mri": ["mri", "mr "],
    "ct": ["ct", "computed tomography"],
    "xray": ["xray", "x-ray", "radiograph"],
    "us": ["ultrasound", "us "],
    "pet": ["pet"],
}

ANATOMY = {
    "brain": ["brain", "head", "intracranial", "stroke"],
    "chest": ["chest", "lung", "pulmonary", "thorax"],
    "abdomen": ["abdomen", "abdominal", "pelvis", "renal", "kidney", "liver"],
    "spine": ["spine", "cervical", "thoracic", "lumbar"],
    "knee": ["knee"],
    "shoulder": ["shoulder"],
    "hip": ["hip"],
    "ankle": ["ankle"],
    "foot": ["foot"],
    "hand": ["hand", "wrist"],
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def contains_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def get_matches(text: str, dictionary: dict[str, list[str]]) -> set[str]:
    text = normalize(text)
    return {key for key, terms in dictionary.items() if contains_any(text, terms)}


def predict_relevance(case):
    current_text = normalize(case.current_study.study_description)

    prior_texts = [
        normalize(study.study_description)
        for study in case.prior_studies
    ]

    if not prior_texts:
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

    current_modalities = get_matches(current_text, MODALITIES)
    current_anatomy = get_matches(current_text, ANATOMY)

    results = []

    for study, prior_text, sim in zip(case.prior_studies, prior_texts, sims):
        prior_modalities = get_matches(prior_text, MODALITIES)
        prior_anatomy = get_matches(prior_text, ANATOMY)

        score = float(sim)

        if current_modalities and prior_modalities and current_modalities & prior_modalities:
            score += 0.05

        if current_anatomy and prior_anatomy and current_anatomy & prior_anatomy:
            score += 0.08

        if current_anatomy and prior_anatomy and not (current_anatomy & prior_anatomy):
            score -= 0.08

        is_relevant = score > THRESHOLD
        results.append((study.study_id, bool(is_relevant)))

    return results