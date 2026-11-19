## [Which Condition Fits? Multi-Label Diagnosis with Supporting Evidence from Symptom Description]

Multi-label text classification for distinguishing commonly confused mental health conditions (ADHD, anxiety, depression) from self-reported symptom descriptions, with extractive rationales highlighting the phrases that support each prediction.


## Quick start (reviewers)

Needs Python 3.10 or newer (3.12 or 3.13 reproduces the reported hashes exactly) and bash (Linux, macOS,
WSL, or Git Bash on Windows). On Ubuntu/WSL, first: `sudo apt install python3-venv`.

```bash
git clone https://github.com/eduardokdp/Which-Condition-Fits-.git
cd Which-Condition-Fits-/proposed-solution
bash quickstart.sh      # Milestone 3: setup, 26 tests, pipeline, hash check, figures, one prediction (~2 min)
bash reproduce_m4.sh    # Milestone 4: 5-fold CV grid for 7 methods, one test evaluation, figures, tables (~3 min)
```

Both scripts also work from the repository root (shortcuts with the same names); they always run inside
`proposed-solution/`, create `proposed-solution/.venv`, and install the packages themselves. A successful run ends with:

```
== reproducibility check PASSED (Milestone 3 quickstart): identical to the reported run (6371861ba793...)
== reproducibility check PASSED (m4_sample): identical to the reported run (c8c2b0fd9803...)
```

Manual installation, VS Code tasks and troubleshooting: [`proposed-solution/README.md`](proposed-solution/README.md)
(Sections 5 and 6). Milestone 4 report and how its tables and figures are produced: [`experiments/`](experiments/).

## Overview

Most existing approaches to mental health classification from text treat diagnosis as single-label, one condition versus healthy controls, or one condition among several. In practice, conditions like ADHD, anxiety, and depression share overlapping symptoms (e.g., restlessness, inattention, irritability), which makes single-label framing a poor match for how misdiagnosis actually happens.

This project proposes a multi-label classifier that:
- Predicts multiple plausible conditions from a single self-reported description, not just one
- Extracts the specific phrases that support each predicted condition (rationale extraction)
- Is evaluated specifically on its ability to separate commonly confused conditions, rather than only distinguishing a condition from a healthy control

## Key Features

- **Multi-label classification:** Allows multiple conditions to be predicted for one symptom description.
- **Evidence extraction:** Highlights text segments that contribute to each condition prediction.
- **Baseline comparison:** Compares the proposed approach against alternative classification methods.
- **Model evaluation:** Examines overall performance, per-condition results, confusion between conditions, and evidence quality.

## Project Approach

The project was developed across four milestones:

1. **Problem Definition:** Identified the challenges of distinguishing overlapping mental health symptoms and established the project's motivation and objectives.
2. **Literature Review:** Examined existing approaches to mental health classification, related research, and gaps in current methods.
3. **Proposed Solution:** Developed the proposed classification approach, implemented the supporting code, and evaluated its performance.
4. **Evaluation:** Compared the method with six baselines under one cross-validated protocol, studied its hyperparameters, and checked where it helps (report in `experiments/`, reproduced by `bash reproduce_m4.sh`).

## Technical Implementation

The implementation is organized into separate modules for data preparation, model processing, classification, evidence extraction, and evaluation.

The main components include:

- **Data preparation:** Loading, cleaning, and preparing symptom descriptions and their associated labels.
- **Feature processing:** Extracting useful representations and features from text.
- **Classification:** Producing condition predictions using a multi-label classification approach.
- **Evidence extraction:** Identifying relevant text segments associated with individual predictions.
- **Evaluation:** Measuring model performance and comparing results against baseline approaches.

For more details, see the `proposed-solution/` directory.

## Team

| Name | Milestone 1: Proposal | Milestone 2: Literature Review | Milestone 3: Proposed-Solution | Milestone 4: Evaluation |
|---|---|---|---|---|
| Niyaaz Baines | Problem definition (1.1) | History, Abstract | Formal definition, Abstract | Research questions (4.2), Introduction |
| Eduardo Kallina de Paula | Motivation, motivating example, gaps in existing solutions (1.2–1.2.2) | Introduction, Positioning | Proposed Method, Algorithm, References | Baselines (4.4) |
| Thiago Vasconcelos Pinheiro Lopes | Formal problem definition, running example (1.3–1.3.1) | Hierarchy, Taxonomy | Pseudo-code, Implementation | Benchmark dataset (4.3) |
| Priyanshkumar Ghanshyambhai Patel | Abstract | Comparative Analysis, Table | Figure 1, Integration | Evaluation strategy and metrics (4.5), experiments, results |

## Repository Structure

```
.
├── proposal/                         # Milestone 1: Problem definition and proposal
├── literature-review/                # Milestone 2: Background and related research
├── proposed-solution/                 # Milestone 3: Implementation and evaluation
│   ├── data/                          # Dataset and data-related files
│   ├── src/                           # Source code
│   │   ├── cmn/                       # Shared utilities and configuration
│   │   ├── core/                      # Core model and prediction logic
│   │   ├── dataprep/                  # Data loading and preprocessing
│   │   └── eval/                      # Evaluation methods and metrics
│   ├── scripts/                       # Data, evaluation, and visualization scripts
│   ├── tests/                         # Tests
│   ├── evaluation/                    # Evaluation data, results, and reports
│   ├── requirements.txt               # Python dependencies, compatible versions (Python 3.10+)
│   ├── requirements-lock.txt          # Exact versions of the reported runs (Python 3.12/3.13)
│   ├── requirements-viz.txt           # Visualization dependencies
│   ├── CoRE.code-workspace            # VS Code workspace with ready-made tasks
│   ├── quickstart.sh                  # Milestone 3 quick start
│   └── reproduce_m4.sh                # Milestone 4 reproduction (runs scripts/reproduce_m4.sh)
├── experiments/                       # Milestone 4: Evaluation report (main.tex, PDF, generated tables and figures)
├── quickstart.sh                      # Shortcut: runs proposed-solution/quickstart.sh
├── reproduce_m4.sh                    # Shortcut: runs proposed-solution/scripts/reproduce_m4.sh
└── README.md                          # Project documentation
```

## License

Code: MIT (`proposed-solution/LICENSE`). Data: public Reddit posts redistributed by the GitHub projects listed
in `proposed-solution/README.md` (Section 3), each under its own terms. Research prototype only; not a
diagnostic tool.
