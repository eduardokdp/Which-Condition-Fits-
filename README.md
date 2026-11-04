## [Which Condition Fits? Multi-Label Diagnosis with Supporting Evidence from Symptom Description]

Multi-label text classification for distinguishing commonly confused mental health conditions (ADHD, anxiety, depression) from self-reported symptom descriptions, with extractive rationales highlighting the phrases that support each prediction.


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

The project was developed across three milestones:

1. **Problem Definition:** Identified the challenges of distinguishing overlapping mental health symptoms and established the project's motivation and objectives.
2. **Literature Review:** Examined existing approaches to mental health classification, related research, and gaps in current methods.
3. **Proposed Solution:** Developed the proposed classification approach, implemented the supporting code, and evaluated its performance.

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

| Name | Milestone 1: Proposal | Milestone 2: Literature Review | Milestone 3: Proposed-Solution 
|---|---|---|---|
| Niyaaz Baines | Problem definition (1.1) | History, Abstract | Formal definition, Abstract
| Eduardo Kallina de Paula | Motivation, motivating example, gaps in existing solutions (1.2–1.2.2) | Introduction, Positioning | Proposed Method, Algorithm, References
| Thiago Vasconcelos Pinheiro Lopes | Formal problem definition, running example (1.3–1.3.1) | Hierarchy, Taxonomy | Pseudo-code, Implementation
| Priyanshkumar Ghanshyambhai Patel | Abstract | Comparative Analysis, Table | Figure 1, Integration

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
│   ├── requirements.txt               # Python dependencies
│   └── requirements-viz.txt           # Visualization dependencies
└── README.md                          # Project documentation
```
