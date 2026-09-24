# fedxss — Privacy-Preserving Federated Browser-Side XSS Detection

Reproducibility artefact for the MCS dissertation:
*A Privacy-Preserving Framework for Collaborative Browser-Side Cross-Site
Scripting (XSS) Detection Using Client-Side Federated Learning.*

## Setup

```bash
cd ~/Desktop/Thesis/fedxss
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1 — build the dataset (done)

Run from inside `src/`, in this order:

```bash
cd src
python collect.py      # -> data/processed/malicious.csv          (~6,556 payloads)
python benign.py       # -> data/processed/benign.csv             (8,000 records)
python synth.py        # -> data/processed/malicious_records.csv  (6,556 records)
python partition.py    # -> data/processed/train.csv, test.csv, shards/
```

Expected output: 14,556 records total, 45% positive, categories balanced at
roughly 2,185 each, train 11,645 / test 2,911.

Nothing is committed to git — the scripts re-download every source, so the
corpus rebuilds identically from a clean clone.

## Phase 2 onwards (not yet built)

```
features.py     fixed-length feature vector from each observation record
baselines.py    rule-based + centralised supervised        -> Table 4.1 rows 1-2
model.py        shared MLP definition
fed_sim.py      Flower FedAvg simulation                   -> Table 4.1 row 3
dp.py           DP-SGD + RDP accountant, epsilon sweep     -> Tables 4.1, 4.2
external.py     validation on XSS_dataset.csv (independent corpus)
stats.py        Wilcoxon, RM-ANOVA, bootstrap CIs
```

## Chapter 3 corrections this code requires

1. The XSSed archive is defunct; sources are PortSwigger (via SecLists),
   BruteLogic, RSnake, Jhaddix, PayloadsAllTheThings and Juice Shop.
2. Alexa top 10k was retired in 2022; a public top-10,000 domain list is
   used instead.
3. Attack category is assigned by synthesised injection context, not by
   payload string. The contexts are constructed — record this in 5.3.
4. 35% of benign records are hard negatives (legitimate innerHTML /
   document.write / setAttribute use, and benign URLs containing markup).
   Without them the task is trivially separable.
5. Client counts reduced to N in {10, 100}; single MLP architecture.

## Reference material in the parent folder

`FL-IDS/` and `federated_transformer_malicious_url/` are third-party repos
cloned for reference, not authored here. Any code adapted from them must be
attributed in the dissertation.
