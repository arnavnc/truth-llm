import argparse
import json
import random
import time
from typing import Dict, List, Any

import requests

ENDPOINT = "https://query.wikidata.org/sparql"

HEADERS = {
    # WDQS etiquette: identify yourself (helps if you get rate-limited)
    "User-Agent": "truthgeom-mini-project/0.1 (contact: local-run)",
    "Accept": "application/sparql-results+json",
}

PREFIXES = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

QUERIES = {
    "capital": PREFIXES + """
SELECT ?s ?o ?s_en ?s_fr ?o_en ?o_fr WHERE {
  ?s wdt:P31 wd:Q3624078 .          # sovereign state
  ?s wdt:P36 ?o .                  # capital
  FILTER NOT EXISTS { ?s wdt:P36 ?o2 . FILTER(?o2 != ?o) }  # single-valued

  ?s rdfs:label ?s_en FILTER(LANG(?s_en)="en") .
  ?s rdfs:label ?s_fr FILTER(LANG(?s_fr)="fr") .
  ?o rdfs:label ?o_en FILTER(LANG(?o_en)="en") .
  ?o rdfs:label ?o_fr FILTER(LANG(?o_fr)="fr") .
}
LIMIT 600
""",
    "birthplace": PREFIXES + """
SELECT ?s ?o ?s_en ?s_fr ?o_en ?o_fr WHERE {
  ?s wdt:P31 wd:Q5 .               # human
  ?s wdt:P19 ?o .                  # place of birth
  FILTER NOT EXISTS { ?s wdt:P19 ?o2 . FILTER(?o2 != ?o) }  # single-valued

  ?s rdfs:label ?s_en FILTER(LANG(?s_en)="en") .
  ?s rdfs:label ?s_fr FILTER(LANG(?s_fr)="fr") .
  ?o rdfs:label ?o_en FILTER(LANG(?o_en)="en") .
  ?o rdfs:label ?o_fr FILTER(LANG(?o_fr)="fr") .
}
LIMIT 800
""",
    "director": PREFIXES + """
SELECT ?s ?o ?s_en ?s_fr ?o_en ?o_fr WHERE {
  ?s wdt:P31 wd:Q11424 .           # film
  ?s wdt:P57 ?o .                  # director
  FILTER NOT EXISTS { ?s wdt:P57 ?o2 . FILTER(?o2 != ?o) }  # single-valued

  ?s rdfs:label ?s_en FILTER(LANG(?s_en)="en") .
  ?s rdfs:label ?s_fr FILTER(LANG(?s_fr)="fr") .
  ?o rdfs:label ?o_en FILTER(LANG(?o_en)="en") .
  ?o rdfs:label ?o_fr FILTER(LANG(?o_fr)="fr") .
}
LIMIT 800
""",
    "atomic_number": PREFIXES + """
SELECT ?s ?num ?s_en ?s_fr WHERE {
  ?s wdt:P31 wd:Q11344 .           # chemical element
  ?s wdt:P1086 ?num .              # atomic number (numeric)

  ?s rdfs:label ?s_en FILTER(LANG(?s_en)="en") .
  ?s rdfs:label ?s_fr FILTER(LANG(?s_fr)="fr") .
}
LIMIT 200
""",
}

TEMPLATES = {
    "capital": {
        "en": [
            "The capital of {S} is {O}.",
            "{O} is the capital of {S}.",
            "{S}'s capital is {O}.",
        ],
        "fr": [
            "La capitale de {S} est {O}.",
            "{O} est la capitale de {S}.",
            "La capitale de {S} est {O}.",
        ],
    },
    "birthplace": {
        "en": [
            "{S} was born in {O}.",
            "The place of birth of {S} is {O}.",
            "{S}'s place of birth is {O}.",
        ],
        "fr": [
            "Le lieu de naissance de {S} est {O}.",
            "{S} est né(e) à {O}.",
            "Le lieu de naissance de {S} est {O}.",
        ],
    },
    "director": {
        "en": [
            "{S} was directed by {O}.",
            "The director of {S} is {O}.",
            "{O} directed {S}.",
        ],
        "fr": [
            "{S} a été réalisé par {O}.",
            "Le réalisateur de {S} est {O}.",
            "{O} a réalisé {S}.",
        ],
    },
    "atomic_number": {
        "en": [
            "The atomic number of {S} is {O}.",
            "{S} has atomic number {O}.",
            "Atomic number of {S}: {O}.",
        ],
        "fr": [
            "Le numéro atomique de {S} est {O}.",
            "{S} a pour numéro atomique {O}.",
            "Numéro atomique de {S} : {O}.",
        ],
    },
}

def run_sparql(query: str, timeout_s: int = 60, max_retries: int = 5) -> Dict[str, Any]:
    """Run a SPARQL query with simple backoff."""
    for attempt in range(max_retries):
        try:
            r = requests.get(
                ENDPOINT,
                params={"format": "json", "query": query},
                headers=HEADERS,
                timeout=timeout_s,
            )
            if r.status_code == 429:
                # rate limit
                sleep_s = 2 ** attempt
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")

def qid(uri: str) -> str:
    # uri like "http://www.wikidata.org/entity/Q142"
    return uri.rsplit("/", 1)[-1]

def clean_label(x: str) -> str:
    x = x.strip()
    x = " ".join(x.split())
    return x

def sample_records(records: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rng.shuffle(records)
    out = []
    seen = set()
    for r in records:
        # simple dedupe on (S,QID, O/QID)
        key = (r["relation"], r["s_id"], str(r["o_id"]))
        if key in seen:
            continue
        seen.add(key)

        # basic label quality filters
        if len(r["s_en"]) < 2 or len(r["s_fr"]) < 2:
            continue
        if len(r["s_en"]) > 60 or len(r["s_fr"]) > 60:
            continue
        if r["relation"] != "atomic_number":
            if len(r["o_en"]) < 2 or len(r["o_fr"]) < 2:
                continue
            if len(r["o_en"]) > 60 or len(r["o_fr"]) > 60:
                continue

        out.append(r)
        if len(out) >= n:
            break
    return out

def make_false_object(objects: List[Dict[str, Any]], true_obj_id: str, rng: random.Random) -> Dict[str, Any]:
    # For functional relations, any different object makes a false statement.
    # Pick until we find a different one.
    while True:
        cand = rng.choice(objects)
        if cand["o_id"] != true_obj_id:
            return cand

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_relation", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/wikidata_enfr_facts_v1.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    all_examples = []
    for rel, query in QUERIES.items():
        print(f"\n[Fetch] relation={rel}")
        data = run_sparql(query)
        bindings = data["results"]["bindings"]

        records = []
        for b in bindings:
            if rel == "atomic_number":
                s_uri = b["s"]["value"]
                s_en = clean_label(b["s_en"]["value"])
                s_fr = clean_label(b["s_fr"]["value"])
                num = b["num"]["value"]
                records.append({
                    "relation": rel,
                    "s_id": qid(s_uri),
                    "o_id": str(num),   # numeric
                    "s_en": s_en,
                    "s_fr": s_fr,
                    "o_en": str(num),
                    "o_fr": str(num),
                })
            else:
                s_uri = b["s"]["value"]
                o_uri = b["o"]["value"]
                s_en = clean_label(b["s_en"]["value"])
                s_fr = clean_label(b["s_fr"]["value"])
                o_en = clean_label(b["o_en"]["value"])
                o_fr = clean_label(b["o_fr"]["value"])
                records.append({
                    "relation": rel,
                    "s_id": qid(s_uri),
                    "o_id": qid(o_uri),
                    "s_en": s_en,
                    "s_fr": s_fr,
                    "o_en": o_en,
                    "o_fr": o_fr,
                })

        print(f"  candidates: {len(records)}")
        picked = sample_records(records, args.n_per_relation, seed=args.seed + hash(rel) % 10_000)
        print(f"  picked: {len(picked)} (target {args.n_per_relation})")

        # Precompute object pool for false sampling
        obj_pool = [{"o_id": r["o_id"], "o_en": r["o_en"], "o_fr": r["o_fr"]} for r in picked]

        for i, r in enumerate(picked):
            false_obj = make_false_object(obj_pool, r["o_id"], rng)

            # choose templates
            t_en = rng.choice(TEMPLATES[rel]["en"])
            t_fr = rng.choice(TEMPLATES[rel]["fr"])

            eng_true  = t_en.format(S=r["s_en"], O=r["o_en"])
            eng_false = t_en.format(S=r["s_en"], O=false_obj["o_en"])

            fr_true   = t_fr.format(S=r["s_fr"], O=r["o_fr"])
            fr_false  = t_fr.format(S=r["s_fr"], O=false_obj["o_fr"])

            all_examples.append({
                "relation": rel,
                "s_id": r["s_id"],
                "o_true_id": r["o_id"],
                "o_false_id": false_obj["o_id"],
                "eng_true": eng_true,
                "eng_false": eng_false,
                "fr_true": fr_true,
                "fr_false": fr_false,
                "labels": {
                    "s_en": r["s_en"], "s_fr": r["s_fr"],
                    "o_en": r["o_en"], "o_fr": r["o_fr"],
                }
            })

        # be polite to WDQS
        time.sleep(1.0)

    # write dataset
    import os
    os.makedirs("data", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in all_examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_examples)} examples to {args.out}")
    # print one example
    print("Example:", all_examples[0])

if __name__ == "__main__":
    main()
