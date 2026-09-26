import csv
import re
import unicodedata
from collections import defaultdict

BASE = "student_resource/dataset/train"

def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm2(s):
    s = norm(s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def read_source(path):
    rows = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            eid = r["entity_id"]
            rows[eid] = {
                "name": norm2(r["business_name"]),
                "address": norm2(r["business_address"]),
                "country": norm2(r["country"]),
            }
    return rows

print("Loading sources...")

s1 = read_source(f"{BASE}/train_source1.tsv")
s2 = read_source(f"{BASE}/train_source2.tsv")
s3 = read_source(f"{BASE}/train_source3.tsv")

print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")

# Build lookup indices.
indices = {}

for label, source in [("s2", s2), ("s3", s3)]:
    by_name = defaultdict(set)
    by_address = defaultdict(set)
    by_pair = defaultdict(set)

    for eid, r in source.items():
        if r["name"]:
            by_name[(r["country"], r["name"])].add(eid)

        if r["address"]:
            by_address[(r["country"], r["address"])].add(eid)

        if r["name"] and r["address"]:
            by_pair[(r["country"], r["name"], r["address"])].add(eid)

    indices[label] = (by_name, by_address, by_pair)

# Read ground truth.
truth = {}

with open(f"{BASE}/train_ground_truth.tsv", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")

    for r in reader:
        truth[r["source1_entity_id"]] = set(
            x.strip()
            for x in r["matched_entity_ids"].split(",")
            if x.strip()
        )

print(f"Ground truth S1 rows: {len(truth):,}")

# entity_id -> source based on prefix is NOT assumed.
# Determine membership directly.
s2_ids = set(s2)
s3_ids = set(s3)

def evaluate(mode):
    total_true = 0
    recovered = 0
    s1_with_true = 0
    s1_with_any_prediction = 0

    for i, (eid, r1) in enumerate(s1.items(), 1):
        predicted = set()

        for label, source_ids in [("s2", s2_ids), ("s3", s3_ids)]:
            by_name, by_address, by_pair = indices[label]

            if mode == "name":
                predicted |= by_name.get(
                    (r1["country"], r1["name"]), set()
                )

            elif mode == "address":
                predicted |= by_address.get(
                    (r1["country"], r1["address"]), set()
                )

            elif mode == "pair":
                predicted |= by_pair.get(
                    (r1["country"], r1["name"], r1["address"]), set()
                )

            elif mode == "name_or_address":
                predicted |= by_name.get(
                    (r1["country"], r1["name"]), set()
                )
                predicted |= by_address.get(
                    (r1["country"], r1["address"]), set()
                )

        true = truth.get(eid, set())

        total_true += len(true)
        recovered += len(predicted & true)

        if true:
            s1_with_true += 1
        if predicted:
            s1_with_any_prediction += 1

    recall = recovered / total_true if total_true else 0

    print()
    print(f"=== {mode} ===")
    print(f"True links recovered: {recovered:,} / {total_true:,}")
    print(f"Link recall: {recall:.4%}")
    print(f"S1 with predictions: {s1_with_any_prediction:,}")

for mode in ["name", "address", "pair", "name_or_address"]:
    evaluate(mode)
