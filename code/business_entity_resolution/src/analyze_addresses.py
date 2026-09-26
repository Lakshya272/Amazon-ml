import csv
import re
from collections import Counter

print("Analyzing address patterns in Source 1...")
with open("student_resource/dataset/train/train_source1.tsv", "r", encoding="utf-8", errors="replace") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader, None)
    rows = [next(reader) for _ in range(50000)]

nums = []
for r in rows:
    if len(r) > 2:
        addr = r[2]
        m = re.findall(r"\b\d+\b", addr)
        if m:
            nums.append(m)

print(f"Rows inspected: {len(rows)}, with numbers: {len(nums)}")
print("Sample address numbers:")
for r in rows[:15]:
    if len(r) > 2:
        m = re.findall(r"\b\d+\b", r[2])
        print(f"  [{r[3]}] {r[2][:70]} -> {m}")
