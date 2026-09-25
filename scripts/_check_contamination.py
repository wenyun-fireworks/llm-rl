"""Check whether the eval sets leaked into the training set.

DAPO-Math-17k is assembled from competition math, and AIME 2024 predates it, so
overlap is plausible and would invalidate the headline metric. Exact match is not
enough: the same problem is often reworded or reformatted between sources, so this
does a cheap word-level Jaccard screen and then a character-level similarity on the
best candidates.
"""

import re
from difflib import SequenceMatcher

from llm_rl.data import load_examples

WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


print("loading DAPO-Math-17k...")
train = load_examples("BytedTsinghua-SIA/DAPO-Math-17k", shuffle_seed=0)
train_norm = [normalize(e.problem) for e in train]
train_words = [words(e.problem) for e in train]
train_exact = {t: i for i, t in enumerate(train_norm)}
print(f"  {len(train)} unique training problems")

evals = {
    "aime_2024": load_examples("HuggingFaceH4/aime_2024", split="train", deduplicate=False),
    "aime_2025": load_examples("yentinglin/aime_2025", split="train", deduplicate=False),
    "aime_2026": load_examples("MathArena/aime_2026", split="train", deduplicate=False),
    "math_500": load_examples("HuggingFaceH4/MATH-500", split="test", deduplicate=False),
}

for name, examples in evals.items():
    exact = 0
    near: list[tuple[float, str, str]] = []
    for ex in examples:
        norm = normalize(ex.problem)
        if norm in train_exact:
            exact += 1
            near.append((1.0, ex.problem, train[train_exact[norm]].problem))
            continue
        # Jaccard screen first; SequenceMatcher over 17k candidates is far too slow.
        ew = words(ex.problem)
        if not ew:
            continue
        best_score, best_index = 0.0, -1
        for i, tw in enumerate(train_words):
            inter = len(ew & tw)
            if inter == 0:
                continue
            j = inter / len(ew | tw)
            if j > best_score:
                best_score, best_index = j, i
        if best_index >= 0 and best_score > 0.5:
            ratio = SequenceMatcher(None, norm, train_norm[best_index]).ratio()
            if ratio > 0.8:
                near.append((ratio, ex.problem, train[best_index].problem))

    print(f"\n{name}: {len(examples)} problems | exact matches: {exact} | near-duplicates (>0.8): {len(near) - exact}")
    for ratio, evaltext, traintext in sorted(near, reverse=True)[:3]:
        print(f"  similarity {ratio:.3f}")
        print(f"    eval : {evaltext[:130]}")
        print(f"    train: {traintext[:130]}")
