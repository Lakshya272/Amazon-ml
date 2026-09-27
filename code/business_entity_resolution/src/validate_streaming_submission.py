#!/usr/bin/env python3
"""Memory-bounded integrity check for the two full submission TSVs.

Use after the official validator if its all-candidate in-memory check exceeds RAM.
The inference script emits S1 rows in test-file order, which this checks.
"""

import argparse
import csv
from itertools import zip_longest


def rows(path):
    handle = open(path, encoding="utf-8", newline="")
    try:
        yield from csv.reader(handle, delimiter="\t")
    finally:
        handle.close()


def parse_ids(cell, label, row_number):
    ids = cell.split(",") if cell else []
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} row {row_number}: duplicate target IDs")
    if any(not eid.startswith(("S2-", "S3-")) for eid in ids):
        raise ValueError(f"{label} row {row_number}: invalid target ID prefix")
    return set(ids)


def run(args):
    source = rows(args.source1)
    candidate = rows(args.candidate)
    matching = rows(args.matching)
    if next(source) != ["entity_id", "business_name", "business_address", "country"]:
        raise ValueError("Unexpected test Source 1 header")
    if next(candidate) != ["source1_entity_id", "candidate_entity_ids"]:
        raise ValueError("Unexpected candidate file header")
    if next(matching) != ["source1_entity_id", "matched_entity_ids"]:
        raise ValueError("Unexpected matching file header")
    count = pairs = predicted = 0
    for count, (s, c, m) in enumerate(zip_longest(source, candidate, matching), start=1):
        if s is None or c is None or m is None:
            raise ValueError(f"Files have different row counts near data row {count}")
        if len(s) != 4 or len(c) != 2 or len(m) != 2:
            raise ValueError(f"Malformed TSV at data row {count}")
        if c[0] != s[0] or m[0] != s[0]:
            raise ValueError(f"Source 1 ID or row order mismatch at data row {count}")
        candidates = parse_ids(c[1], "candidate", count)
        matches = parse_ids(m[1], "matching", count)
        if not matches <= candidates:
            raise ValueError(f"Predicted match absent from candidates at data row {count}")
        pairs += len(candidates)
        predicted += len(matches)
        if count % 100000 == 0:
            print(f"Checked {count:,} S1 rows and {pairs:,} candidates", flush=True)
    if count != args.expected_s1:
        raise ValueError(f"Expected {args.expected_s1:,} S1 rows, found {count:,}")
    print(f"PASS: {count:,} S1 rows; {pairs:,} candidates; {predicted:,} matches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", default="student_resource/dataset/test/test_source1.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--expected-s1", type=int, default=1732544)
    run(parser.parse_args())
