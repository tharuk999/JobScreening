# JobScreening
# AI Resume Screening

A machine learning tool that predicts candidate hire/reject outcomes from resume data and generates a fit score for a given job role. Built with scikit-learn and trained on structured resume records.

---

## Overview

This project trains a classification model on historical recruiter decisions to evaluate how well a candidate's resume matches a target role. It encodes resume attributes (skills, education, certifications, experience) into a numerical feature vector and outputs a probability-based fit score between 0 and 100.

The system also produces three visualizations to support analysis of the underlying dataset.

---

## Features

- Logistic regression classifier with balanced class weighting
- One-hot encoding of categorical resume fields
- Fit score output (probability of hire, expressed as a percentage)
- Confusion matrix and classification report for model evaluation
- Hire rate charts by job role and education level
- Skill frequency comparison between hired and rejected candidates
- Interactive command-line interface for manual resume input

---

## Requirements

**Python 3.8 or higher**

Install dependencies with:

```bash
pip install numpy matplotlib scikit-learn
```

---

## Dataset

The model expects a CSV file named `AI_Resume_Screening.csv` in the project root directory.

Required columns:

| Column | Description |
|---|---|
| `Skills` | Comma-separated list of skills |
| `Experience (Years)` | Integer years of experience |
| `Education` | Highest education level |
| `Certifications` | Single certification or "None" |
| `Job Role` | Target job role |
| `Recruiter Decision` | "Hire" or "Reject" |

---

## Supported Values

The model recognizes the following categorical values. Any input outside these lists is ignored.

**Skills**
TensorFlow, NLP, Pytorch, Deep Learning, Machine Learning, Python, SQL, Ethical Hacking, Cybersecurity, Linux, React, Java, Networking, C++

**Education**
B.Sc, MBA, B.Tech, PhD, M.Tech

**Certifications**
None, Google ML, Deep Learning Specialization, AWS Certified

**Job Roles**
AI Researcher, Data Scientist, Cybersecurity Analyst, Software Engineer

---

## Usage

Run the script from the project directory:

```bash
python main.py
```

On startup, the script will:

1. Load and encode the dataset
2. Train the model on 80% of the data
3. Evaluate performance on the held-out 20%
4. Generate and save three visualization charts
5. Launch an interactive fit score calculator

In the calculator, you will be prompted to select a job role, enter your skills, years of experience, education level, and certification. The model returns a fit score and a short assessment.

---

## Output Files

| File | Description |
|---|---|
| `confusion_matrix.png` | Predicted vs. actual hire/reject outcomes on the test set |
| `hire_rates.png` | Hire rate breakdown by job role and education level |
| `skill_hire_rates.png` | Skill frequency comparison between hired and rejected candidates |

---

## Model Details

**Algorithm:** Logistic Regression (L-BFGS solver)

**Preprocessing:** StandardScaler - features are normalized to zero mean and unit variance before training and inference. The scaler is fit exclusively on training data to prevent data leakage.

**Class weighting:** `balanced` - automatically adjusts loss weights inversely proportional to class frequency, compensating for imbalanced hire/reject ratios in the dataset.

**Train/test split:** 80/20, stratified by label to preserve class distribution in both sets.

**Feature count:** 38 features - 14 skill flags, 1 experience value, 5 education flags, 4 certification flags, 4 job role flags.

---

## Fit Score Interpretation

| Score Range | Assessment |
|---|---|
| 75% and above | Strong match |
| 50% to 74% | Moderate match - some gaps to address |
| Below 50% | Low match - consider upskilling |

The fit score is the raw probability output of `predict_proba`, not a hard classification threshold. This means a score of 60% does not guarantee a hire prediction - it reflects the model's confidence level given the input features.

---

## Project Structure

```
.
├── resume_screening.py       # Main script
├── AI_Resume_Screening.csv   # Training dataset (required)
├── confusion_matrix.png      # Generated on run
├── hire_rates.png            # Generated on run
└── skill_hire_rates.png      # Generated on run
```

---

## Limitations

- The model is constrained to the four job roles and fixed skill vocabulary defined in the script. Inputs outside these categories are ignored.
- Performance is dependent on the quality and size of the training dataset. A small or biased dataset will produce unreliable predictions.
- Logistic regression assumes a linear decision boundary. If hire/reject patterns in the data are non-linear.
- The fit score reflects patterns in historical recruiter decisions, which may themselves contain bias.