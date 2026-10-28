"""Deterministic filler text for long prompts in the serve tools."""
import random

_NOUNS = ["river", "engine", "garden", "lantern", "harbor", "violin", "glacier",
          "market", "orchard", "satellite", "library", "canyon", "bakery"]
_VERBS = ["follows", "measures", "remembers", "outlines", "repairs", "paints",
          "describes", "carries", "observes", "records"]
_ADJECTIVES = ["quiet", "ancient", "bright", "narrow", "busy", "distant",
               "patient"]


def filler(sentences, seed):
    """`sentences` numbered notes (~13 tokens each), the same for a seed."""
    rng = random.Random(seed)
    return " ".join(
        f"Note {i}: the {rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)} "
        f"{rng.choice(_VERBS)} the {rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)}."
        for i in range(sentences))
