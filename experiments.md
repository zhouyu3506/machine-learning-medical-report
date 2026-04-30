# Experiments

## Baseline
I implemented a TF-IDF based relevance classifier. For each case, I vectorized the current study description and all prior study descriptions using TF-IDF with unigram and bigram features. I then computed cosine similarity between the current study and each prior study.

A prior study is predicted as relevant if its cosine similarity score is above a fixed threshold.

## What worked
- TF-IDF is fast and reliable for API evaluation.
- It works well when the current and prior studies share important keywords such as modality or anatomy.
- It avoids external API calls and reduces timeout risk.

## What failed
- It may miss semantic matches when different medical terms are used.
- It does not understand deeper clinical context.
- The threshold may not generalize perfectly to the hidden split.

## Next-step improvements
- Replace TF-IDF with a sentence embedding model such as MiniLM once deployment stability is solved.
- Add rule-based features for modality, anatomy, and recency.
- Tune the threshold on the public evaluation set.
- Add a medical-domain reranker for borderline cases.