# fedxss: Privacy-Preserving Federated Browser-Side XSS Detection

This is the reproducibility artefact for the MCS dissertation:

> *A Privacy-Preserving Framework for Collaborative Browser-Side Cross-Site
> Scripting (XSS) Detection Using Client-Side Federated Learning.*

The repository covers the whole experimental pipeline. It builds the corpus,
extracts features, and trains the detector. It then runs the federated
simulation and the differential-privacy (DP) sweep, validates on an external
corpus, and produces the statistical analysis. It also includes an
inference-only Chrome extension that runs the exported model inside the
browser.

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
- [Repository structure](#repository-structure)
- [Experimental design](#experimental-design)
- [Results](#results)
- [Browser extension](#browser-extension)
- [Deviations from the Chapter 3 methodology](#deviations-from-the-chapter-3-methodology)
- [Data sources and attribution](#data-sources-and-attribution)

## Overview

Each browser acts as a federated client. It turns browser-observed events
(URL, script content, DOM sink and origin) into a fixed-length feature
vector and trains a small MLP on them locally. Only model updates are shared,
and FedAvg combines them. DP-SGD with per-example clipping and Rényi DP
accounting can be switched on to bound what those updates reveal about any
single training example.

The study measures:

- **RQ1**: how federated detection compares with a centralised upper bound.
- **RQ3**: how much detection quality is lost at different privacy budgets (ε).
- **RQ4**: how robust the model is to non-IID client data and to partial
  client participation.

## Requirements

| Component | Version | Purpose |
|---|---|---|
| Python | 3.14 (the version used for development and evaluation) | Full pipeline |
| Python packages | see [requirements.txt](requirements.txt) | NumPy, pandas, scikit-learn, SciPy, Flower, dp-accounting |
| Node.js | any recent LTS | Only for the feature parity check in `tools/` |
| Google Chrome | MV3-capable | Only for the browser extension |

The detector is written in pure NumPy so that it can be ported directly to
browser JavaScript. PyTorch and Opacus are deliberately not dependencies:
`src/dp.py` implements per-example clipping itself and uses Google's
`dp-accounting` for the privacy accounting. The rationale is set out in
[requirements.txt](requirements.txt) and [src/model.py](src/model.py).

**Optional external dataset:** `external.py` needs `XSS_dataset.csv`, a
public XSS dataset from Kaggle (see
[Data sources and attribution](#data-sources-and-attribution)). Download it
and place it in the directory above the repository root. Phases 1–4 do not
need it.

## Installation

```bash
git clone <repository-url> fed-xss
cd fed-xss
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

Run every script from inside `src/`, in the order below. Phase 1 downloads
all of its sources again, so a clean clone rebuilds exactly the same corpus.

```bash
cd src

# Phase 1: dataset construction
python collect.py        # -> data/processed/malicious.csv          (~6,556 payloads)
python benign.py         # -> data/processed/benign.csv             (8,000 records)
python synth.py          # -> data/processed/malicious_records.csv  (6,556 records)
python partition.py      # -> data/processed/{train,test}.csv, shards/

# Phase 2: features and baselines
python baselines.py      # -> results/baselines.json   (Table 4.1, rows 1–2)

# Phase 3: federated learning without DP
python fed_sim.py        # -> results/fed_nodp.json    (Table 4.1, row 3; RQ4 panel)

# Phase 4: differential privacy
python dp_sweep.py       # -> results/dp_sweep.json    (Table 4.2; Figure 4.1)

# Phase 5: external validation, statistics and tables
python external.py       # -> results/external.json
python stats.py          # -> results/stats.json
python report_tables.py  # -> results/tables.txt

# Phase 6: browser client
python export_model.py   # -> extension/model.json
```

**Configuration.** `partition.py` accepts `--seed` (default `42`) and
`--clients` (default `10 100`). Every other script is configured through
constants at the top of the file.

**Committed outputs.** The `results/` directory is under version control, so
you can check the Chapter 4 figures without re-running anything.
`data/processed/` is not committed, because Phase 1 regenerates it.

## Repository structure

```
fed-xss/
├── src/                  Experimental pipeline (see module map below)
├── extension/            Inference-only MV3 Chrome extension
├── tools/                Python/JavaScript feature-parity checks
├── results/              Committed experiment outputs (JSON + tables.txt)
├── data/                 Generated corpus (not committed)
└── requirements.txt
```

### Module map (`src/`)

| Module | Phase | Responsibility |
|---|---|---|
| `schema.py` | — | Data model for a browser-observed event: URL, script, sink, origin, category, label, family |
| `contexts.py` | — | Injection-context builders shared by the benign and malicious classes |
| `net.py` | — | HTTPS fetching with a certifi/system-store fallback; TLS verification is never turned off |
| `collect.py` | 1a | Collects, normalises and de-duplicates payloads |
| `benign.py` | 1b | Builds benign records from real page markup (v3; the file header records earlier versions) |
| `synth.py` | 1c | Turns each payload into an observed browser event |
| `partition.py` | 1d | Grouped train/test split and Dirichlet client sharding |
| `features.py` | 2a | Stateless 320-dimensional feature vector using hashed character 3-grams |
| `model.py` | 2 | Two-layer NumPy MLP with closed-form per-example gradients |
| `baselines.py` | 2b | Rule-based filter and centralised supervised upper bound |
| `fed_core.py` | 3 | Framework-independent FedAvg; the single point where DP plugs in |
| `fed_sim.py` | 3 | Experiment grid without DP |
| `dp.py` | 4 | DP-SGD with per-example clipping and Rényi DP accounting |
| `dp_sweep.py` | 4 | Privacy/utility sweep across target ε |
| `external.py` | 5 | Validation on an independent corpus (`XSS_dataset.csv`) |
| `stats.py` | 5 | Wilcoxon and Friedman tests, bootstrap confidence intervals, instability reporting |
| `report_tables.py` | 5 | Condenses `results/*.json` into `results/tables.txt` |
| `export_model.py` | 6 | Exports the weights and feature layout for the extension |

## Experimental design

| Factor | Levels |
|---|---|
| Data heterogeneity | `iid`, `cat_mild`, `cat_severe`, `lab_mild`, `lab_severe` |
| Number of clients | 10, 100 |
| Client participation | 1.0 (full), 0.2 (partial) |
| Random seeds | 10 |
| DP target ε | none, 10, 5, 2, 1 (δ = 1e-5, clipping norm = 1.0) |

**Heterogeneity.** Non-IID shards come from a Dirichlet distribution over
either attack **category** (`cat_*`) or class **label** (`lab_*`), with
α = 1.0 for mild skew and α = 0.1 for severe skew. Category skew alone had no
measurable effect. This is because the features that separate attacks from
benign traffic are the same for reflected, stored and DOM-based XSS. Label
skew is therefore the stricter test, and both are reported.

**Training.** 30 communication rounds, 5 local epochs, learning rate 0.3,
batch size 64, 64 hidden units.

**Model size.** 20,609 parameters. Each client uploads 80.5 KiB per round.

## Results

**Dataset.** 14,556 records. The training set has 11,645 records (44.9%
malicious) and the test set has 2,911 (45.7% malicious).

### Detection performance (RQ1, RQ4)

| Configuration | F1 | FPR |
|---|---|---|
| Rule-based filter | 0.8119 | 0.3818 |
| Centralised supervised (upper bound) | 0.9951 ± 0.0003 | 0.0000 |
| Federated, IID, 10 clients, full participation | 0.9944 ± 0.0003 | 0.0003 |
| Federated, IID, 100 clients, full participation | 0.9886 ± 0.0001 | 0.0000 |
| Federated, `lab_severe`, 10 clients, 20% participation | 0.8483 ± 0.1687 | 0.3588 |

With IID data, federated training comes within about 0.001 F1 of centralised
training. A Wilcoxon signed-rank test over 10 paired seeds finds the gap
statistically significant, but it is too small to matter in practice.

The failure mode is severe label skew combined with partial participation and
a small number of clients. In that setting the model does not converge on
every seed. That cell is therefore reported by its median, its range and the
number of runs that failed to converge, because a mean would hide the
instability.

### Cost of privacy (RQ3)

Change in F1 compared with the non-private run of the same data profile:

| ε | IID | `lab_severe` |
|---|---|---|
| 10 | −0.006 | −0.031 |
| 5 | −0.006 | −0.074 |
| 2 | −0.007 | −0.132 |
| 1 | −0.008 | −0.155 |

With IID data, the privacy cost stays negligible all the way down to ε = 1.
Under severe label skew, the cost adds to the effect of heterogeneity and
becomes substantial.

### External validation

The external corpus is `XSS_dataset.csv`, a public dataset from Kaggle:
8,556 records (1,183 benign and 7,373 malicious). Other researchers
assembled it for a different study. It
has no URL, sink or origin metadata, so the detector works from script
content alone.

| Model | F1 | FPR |
|---|---|---|
| Centralised | 0.9973 ± 0.0002 | 0.0068 |
| Federated, IID | 0.9961 ± 0.0007 | 0.0066 |

Performance carries over to this corpus. That answers the concern that the
detector might have learned the quirks of the synthetic generator rather than
XSS itself.

## Browser extension

`extension/` holds an inference-only Manifest V3 Chrome extension. It loads
`model.json` and scores navigation, inline-script and DOM-mutation events on
the device.

- It does **not** train locally.
- It does **not** send any data to a server.

The federated training results come from simulation. An in-browser training
client is future work. For installation steps and the Table 4.3 measurement
harness, see [extension/README.md](extension/README.md).

### Feature parity

`extension/features.js` must produce the same output as `src/features.py`.
After changing either file, run:

```bash
python tools/dump_vectors.py && node tools/parity_check.js
```

Last verified on 300 test records, with a maximum absolute difference of
2.98e-8, which is float32 rounding only.

## Deviations from the Chapter 3 methodology

Building the system required these changes to the methodology set out in
Chapter 3:

1. **Payload sources.** The XSSed archive no longer exists. Payloads come
   instead from PortSwigger (via SecLists), BruteLogic, RSnake, Jhaddix,
   PayloadsAllTheThings and OWASP Juice Shop.
2. **Benign domain list.** Alexa Top Sites was retired in 2022. A public
   top-10,000 domain list replaces it.
3. **Attack category assignment.** Categories come from the synthesised
   injection context, not from the payload string. Because these contexts
   are constructed, this is recorded as a limitation in Section 5.3.
4. **Hard negatives.** 35% of benign records are hard negatives. These
   include legitimate `innerHTML`, `document.write` and `setAttribute` use,
   and benign URLs that contain markup. Benign values are real page markup
   placed in the same contexts as the malicious ones. Earlier versions
   scored F1 ≈ 1.0000 because the label could be read from where a record
   came from rather than from its content. The history is recorded in
   `src/benign.py`.
5. **Scope.** Client counts are reduced to N ∈ {10, 100}, and a single MLP
   architecture is used.
6. **Seeds.** Increased from 5 to 10. With 5 paired seeds, the smallest
   two-sided p-value a Wilcoxon signed-rank test can return is 0.0625, so it
   could never reach p < 0.05 whatever the effect size. With 10 seeds the
   minimum is 0.002.
7. **Statistical test.** Friedman's test replaces repeated-measures ANOVA.
   Under severe skew, per-seed F1 splits into two groups (runs that converge
   and runs that do not), which breaks the ANOVA normality assumption.
   Shapiro–Wilk results are reported as evidence.
8. **DP granularity.** The guarantee is example-level (local) DP, not
   client-level DP. It is accounted with Google's `dp-accounting` rather
   than Flower's client-level `RdpAccountant`. See the header of
   `src/dp.py`.

## Data sources and attribution

- The malicious payloads come from the public collections listed under
  Deviation 1 above. Each one keeps its original licence.
- `XSS_dataset.csv` is a public dataset published on Kaggle by third
  parties: <Kaggle dataset URL>. It is used only for external validation and
  is not redistributed in this repository. Its use is subject to the licence
  on its Kaggle page.
- The dissertation attributes any third-party code adapted from reference
  implementations.
