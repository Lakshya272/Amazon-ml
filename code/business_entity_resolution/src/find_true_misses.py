import csv
import sys
import re
import unicodedata
from collections import defaultdict

print('Loading GT...', flush=True)
truth = {}
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader, None)
    for r in reader:
        if r and len(r) > 1 and r[1].strip():
            truth[r[0].strip()] = {x.strip() for x in r[1].strip().split(',') if x.strip()}

print('Loading S1 (first 10,000 entities)...', flush=True)
s1 = {}
with open('student_resource/dataset/train/train_source1.tsv', 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader, None)
    for r in reader:
        if r and r[0].strip() in truth:
            s1[r[0].strip()] = (r[1].strip(), r[2].strip(), r[3].strip())
            if len(s1) >= 10000:
                break

target_s1_ids = set(s1.keys())
all_target_cands = set()
for sid in target_s1_ids:
    all_target_cands.update(truth[sid])

_DIGIT_SEQ_RE = re.compile(r'\b\d+\b')
_POSTAL_RE = re.compile(r'\b\d{5,6}\b')
_STOP_WORDS = {
    'and', 'the', 'of', 'in', 'at', 'on', 'for', 'with', 'to', 'a', 'an',
    'private', 'limited', 'corporation', 'incorporated', 'company', 'llc', 'sarl', 'sas', 'gmbh',
    'pvt', 'ltd', 'corp', 'inc', 'co', 'services', 'enterprises', 'trading', 'solutions', 'associates'
}

def clean_norm(s):
    if not s:
        return ""
    s = unicodedata.normalize('NFKC', s.lower())
    return ' '.join(re.sub(r'[^\w\s]', ' ', s).split())

def legal_norm(s):
    s = clean_norm(s)
    toks = [t for t in s.split() if t not in _STOP_WORDS]
    return ' '.join(toks)

idx_a = defaultdict(list)
idx_b = defaultdict(list)
idx_c = defaultdict(list)
idx_e = defaultdict(list)
idx_f = defaultdict(list)
idx_g = defaultdict(list)
cand_raw = {}

print('Indexing S2/S3 for target candidates...', flush=True)
for path in ['student_resource/dataset/train/train_source2.tsv', 'student_resource/dataset/train/train_source3.tsv']:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for r in reader:
            if not r or len(r) < 4:
                continue
            eid, name, addr, country = r[0].strip(), r[1].strip(), r[2].strip(), r[3].strip()
            if eid in all_target_cands:
                cand_raw[eid] = (name, addr, country)

            leg_n = legal_norm(name)
            if leg_n:
                idx_a[(country, leg_n)].append(eid)
                snk = ''.join(leg_n.split())[:8]
                if len(snk) >= 5:
                    idx_g[(country, snk)].append(eid)
                toks = [t for t in leg_n.split() if len(t) >= 3]
                for t in toks[:3]:
                    idx_b[(country, t)].append(eid)
                if len(toks) >= 3:
                    idx_e[(country, f'{toks[0]}_{toks[1]}')].append(eid)
                    idx_e[(country, f'{toks[1]}_{toks[2]}')].append(eid)

            clean_a = clean_norm(addr)
            posts = _POSTAL_RE.findall(clean_a)
            digs = _DIGIT_SEQ_RE.findall(clean_a)
            if posts:
                idx_c[(country, 'postal', posts[0])].append(eid)
            w = [x for x in clean_a.split() if x not in _STOP_WORDS and not x.isdigit() and len(x) >= 3]
            if digs and w:
                idx_c[(country, 'num_word', f'{digs[0]}_{w[0]}')].append(eid)
            if len(w) >= 2:
                idx_f[(country, 'addr_w1', w[1])].append(eid)

print('Retrieving candidates for S1...', flush=True)
missed_pairs = []
for sid in target_s1_ids:
    s_name, s_addr, s_country = s1[sid]
    leg_n = legal_norm(s_name)
    cands = set()
    if leg_n:
        cands.update(idx_a.get((s_country, leg_n), [])[:30])
        snk = ''.join(leg_n.split())[:8]
        if len(snk) >= 5:
            cands.update(idx_g.get((s_country, snk), [])[:15])
        toks = [t for t in leg_n.split() if len(t) >= 3]
        for t in toks[:3]:
            cands.update(idx_b.get((s_country, t), [])[:15])
        if len(toks) >= 3:
            cands.update(idx_e.get((s_country, f'{toks[0]}_{toks[1]}'), [])[:15])
            cands.update(idx_e.get((s_country, f'{toks[1]}_{toks[2]}'), [])[:15])

    clean_a = clean_norm(s_addr)
    posts = _POSTAL_RE.findall(clean_a)
    digs = _DIGIT_SEQ_RE.findall(clean_a)
    if posts:
        cands.update(idx_c.get((s_country, 'postal', posts[0]), [])[:15])
    w = [x for x in clean_a.split() if x not in _STOP_WORDS and not x.isdigit() and len(x) >= 3]
    if digs and w:
        cands.update(idx_c.get((s_country, 'num_word', f'{digs[0]}_{w[0]}'), [])[:15])
    if len(w) >= 2:
        cands.update(idx_f.get((s_country, 'addr_w1', w[1]), [])[:15])

    for true_mid in truth[sid]:
        if true_mid not in cands:
            missed_pairs.append((sid, true_mid))

print(f'\nTotal missed pairs among sample: {len(missed_pairs)}', flush=True)
print('='*80, flush=True)
print('DETAILED SAMPLE OF MISSED TRUE LINKS (A-G FAILS):', flush=True)
print('='*80, flush=True)
for sid, mid in missed_pairs[:35]:
    if mid in cand_raw:
        s_n, s_a, s_c = s1[sid]
        c_n, c_a, c_c = cand_raw[mid]
        print(f'S1:   [{s_c}] {s_n} || ADDR: {s_a}', flush=True)
        print(f'CAND: [{c_c}] {c_n} || ADDR: {c_a}', flush=True)
        print('-'*60, flush=True)
