import json
import os

# A small, curated list to keep label noise low.
# You can expand this later.
FACTS = [
    {"country_en": "France", "country_fr": "la France", "capital_en": "Paris", "capital_fr": "Paris"},
    {"country_en": "Germany", "country_fr": "l'Allemagne", "capital_en": "Berlin", "capital_fr": "Berlin"},
    {"country_en": "Italy", "country_fr": "l'Italie", "capital_en": "Rome", "capital_fr": "Rome"},
    {"country_en": "Spain", "country_fr": "l'Espagne", "capital_en": "Madrid", "capital_fr": "Madrid"},
    {"country_en": "Portugal", "country_fr": "le Portugal", "capital_en": "Lisbon", "capital_fr": "Lisbonne"},
    {"country_en": "Netherlands", "country_fr": "les Pays-Bas", "capital_en": "Amsterdam", "capital_fr": "Amsterdam"},
    {"country_en": "Belgium", "country_fr": "la Belgique", "capital_en": "Brussels", "capital_fr": "Bruxelles"},
    {"country_en": "Switzerland", "country_fr": "la Suisse", "capital_en": "Bern", "capital_fr": "Berne"},
    {"country_en": "Austria", "country_fr": "l'Autriche", "capital_en": "Vienna", "capital_fr": "Vienne"},
    {"country_en": "Poland", "country_fr": "la Pologne", "capital_en": "Warsaw", "capital_fr": "Varsovie"},
    {"country_en": "Sweden", "country_fr": "la Suède", "capital_en": "Stockholm", "capital_fr": "Stockholm"},
    {"country_en": "Norway", "country_fr": "la Norvège", "capital_en": "Oslo", "capital_fr": "Oslo"},
    {"country_en": "Denmark", "country_fr": "le Danemark", "capital_en": "Copenhagen", "capital_fr": "Copenhague"},
    {"country_en": "Russia", "country_fr": "la Russie", "capital_en": "Moscow", "capital_fr": "Moscou"},
    {"country_en": "China", "country_fr": "la Chine", "capital_en": "Beijing", "capital_fr": "Pékin"},
    {"country_en": "Japan", "country_fr": "le Japon", "capital_en": "Tokyo", "capital_fr": "Tokyo"},
    {"country_en": "India", "country_fr": "l'Inde", "capital_en": "New Delhi", "capital_fr": "New Delhi"},
    {"country_en": "Canada", "country_fr": "le Canada", "capital_en": "Ottawa", "capital_fr": "Ottawa"},
    {"country_en": "Australia", "country_fr": "l'Australie", "capital_en": "Canberra", "capital_fr": "Canberra"},
    {"country_en": "Brazil", "country_fr": "le Brésil", "capital_en": "Brasilia", "capital_fr": "Brasilia"},
]

ENG_TEMPLATE = "The capital of {country} is {capital}."
FR_TEMPLATE  = "La capitale de {country} est {capital}."

def make_pairs(facts):
    pairs = []
    n = len(facts)
    for i, f in enumerate(facts):
        true_cap_en = f["capital_en"]
        true_cap_fr = f["capital_fr"]

        # Deterministic wrong capital: next item in list (ensures plausible, non-nonsense)
        wrong = facts[(i + 1) % n]
        false_cap_en = wrong["capital_en"]
        false_cap_fr = wrong["capital_fr"]

        eng_true = ENG_TEMPLATE.format(country=f["country_en"], capital=true_cap_en)
        eng_false = ENG_TEMPLATE.format(country=f["country_en"], capital=false_cap_en)

        fr_true = FR_TEMPLATE.format(country=f["country_fr"], capital=true_cap_fr)
        fr_false = FR_TEMPLATE.format(country=f["country_fr"], capital=false_cap_fr)

        pairs.append({
            "id": i,
            "eng_true": eng_true,
            "eng_false": eng_false,
            "fr_true": fr_true,
            "fr_false": fr_false,
        })
    return pairs

def main():
    os.makedirs("data", exist_ok=True)
    out_path = "data/capitals_pairs.jsonl"

    pairs = make_pairs(FACTS)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs)} items to {out_path}")
    print("Example row:", pairs[0])

if __name__ == "__main__":
    main()
