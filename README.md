## [Which Condition Fits? Multi-Label Diagnosis with Supporting Evidence from Symptom Description]

Multi-label text classification for distinguishing commonly confused mental health conditions (ADHD, anxiety, depression) from self-reported symptom descriptions, with extractive rationales highlighting the phrases that support each prediction.


## Overview

Most existing approaches to mental health classification from text treat diagnosis as single-label, one condition versus healthy controls, or one condition among several. In practice, conditions like ADHD, anxiety, and depression share overlapping symptoms (e.g., restlessness, inattention, irritability), which makes single-label framing a poor match for how misdiagnosis actually happens.

This project proposes a multi-label classifier that:
- Predicts multiple plausible conditions from a single self-reported description, not just one
- Extracts the specific phrases that support each predicted condition (rationale extraction)
- Is evaluated specifically on its ability to separate commonly confused conditions, rather than only distinguishing a condition from a healthy control


## Team

| Name | Contributions |
|---|---|
| Niyaaz Baines |
| Eduardo Kallina de Paula |
| Thiago Vasconcelos Pinheiro Lopes |
| Priyanshkumar Ghanshyambhai Patel |

## Repository Structure

```
.
├── proposal/            # Milestone 1: Project Definition & Team Formation
├── literature-review/   # Milestone 2: Background & Related Work
├── ...
└── README.md
```