# GITHUB: https://github.com/tharuk999/JobScreening

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

path = "AI_Resume_Screening.csv"
all_skills = ["TensorFlow", "NLP", "Pytorch", "Deep Learning", "Machine Learning", "Python", "SQL", "Ethical Hacking", "Cybersecurity", "Linux", "React", "Java", "Networking","C++"]
all_educations = ["B.Sc", "MBA", "B.Tech", "PhD", "M.Tech"]
all_certs = ["None", "Google ML", "Deep Learning Specialization", "AWS Certified"]
all_jobs = ["AI Researcher", "Data Scientist", "Cybersecurity Analyst", "Software Engineer"]

# PREPROCESSING

def load_data(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["Skills"] = [s.strip() for s in row["Skills"].split(",")]
            row["Experience (Years)"] = int(row["Experience (Years)"])
            rows.append(row)
    return rows

def encode_row(row):
    features = []
    experience_bins = [(0, 2), (3, 5), (6, 8), (9, 10)]

    for skill in all_skills: # SKILLS
        features.append(1 if skill in row["Skills"] else 0)
    exp = row["Experience (Years)"]
    for low, high in experience_bins: # EXPERIENCE
        features.append(1 if low <= exp <= high else 0)
    for edu in all_educations: # EDUCATIONS
        features.append(1 if row["Education"] == edu else 0)
    for cert in all_certs: # CERTIFICATIONS
        features.append(1 if row["Certifications"] == cert else 0)
    for role in all_jobs: # JOBS
        features.append(1 if row["Job Role"] == role else 0)

    return features

def feature_names():
    experience_bins = [(0, 2), (3, 5), (6, 8), (9, 10)]
    names = (
        [f"skill_{s}" for s in all_skills]
        + [f"exp_{low}_{high}yr" for low, high in experience_bins]
        + [f"edu_{e}" for e in all_educations]
        + [f"cert_{c}" for c in all_certs]
        + [f"role_{r}" for r in all_jobs]
    )
    return names

def build_dataset(rows):
    X, y = [], []
    for row in rows:
        X.append(encode_row(row))
        y.append(1 if row["Recruiter Decision"] == "Hire" else 0)
    return np.array(X, dtype=float), np.array(y)

# MODEL TRAINING

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight="balanced", # this is supposed to balance the data, add more weights to the rejected data since there's less of it
        max_iter=1000,
        solver="lbfgs"
    )
    model.fit(X_scaled, y_train)
    return model, scaler

# EVALUATE

def evaluate(model, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    print("\nModel Performance")
    print(classification_report(y_test, y_pred, target_names=["Reject", "Hire"]))

    # Confusion matrix (research this more)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Reject", "Hire"],
        colorbar=False, ax=ax, cmap="Blues"
    )
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved -> confusion_matrix.png")

# VISUALIZATION

def plot_hire_rate_by_role_and_edu(rows):
    """Side-by-side bar charts: hire rate per job role and per education."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # by Job Role
    role_hire = {r: 0 for r in all_jobs}
    role_total = {r: 0 for r in all_jobs}
    for row in rows:
        role = row["Job Role"]
        role_total[role] += 1
        if row["Recruiter Decision"] == "Hire":
            role_hire[role] += 1
    rates_role = [role_hire[r] / role_total[r] * 100 for r in all_jobs]
    axes[0].bar(all_jobs, rates_role, color="#3498db", edgecolor="white")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Hire Rate (%)")
    axes[0].set_title("Hire Rate by Job Role")
    axes[0].tick_params(axis="x", rotation=15)
    for i, v in enumerate(rates_role):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

    # - by Education
    edu_hire = {e: 0 for e in all_educations}
    edu_total = {e: 0 for e in all_educations}
    for row in rows:
        edu = row["Education"]
        edu_total[edu] += 1
        if row["Recruiter Decision"] == "Hire":
            edu_hire[edu] += 1
    rates_edu = [edu_hire[e] / edu_total[e] * 100 for e in all_educations]
    axes[1].bar(all_educations, rates_edu, color="#9b59b6", edgecolor="white")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Hire Rate (%)")
    axes[1].set_title("Hire Rate by Education Level")
    for i, v in enumerate(rates_edu):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("hire_rates.png", dpi=150)
    plt.show()
    print("Saved -> hire_rates.png")

def plot_skill_hire_rates(rows):
    """How often is each skill present among hired vs. rejected candidates?"""
    skill_hire = {s: 0 for s in all_skills}
    skill_reject = {s: 0 for s in all_skills}

    for row in rows:
        bucket = skill_hire if row["Recruiter Decision"] == "Hire" else skill_reject
        for skill in row["Skills"]:
            if skill in bucket:
                bucket[skill] += 1

    total_hire = sum(1 for r in rows if r["Recruiter Decision"] == "Hire")
    total_reject = len(rows) - total_hire

    hire_pct = [skill_hire[s] / total_hire * 100 for s in all_skills]
    reject_pct = [skill_reject[s] / total_reject * 100 for s in all_skills]

    x = np.arange(len(all_skills))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width/2, hire_pct, width, label="Hired", color="#2ecc71")
    ax.bar(x + width/2, reject_pct, width, label="Rejected", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(all_skills, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of candidates with this skill")
    ax.set_title("Skill Frequency: Hired vs Rejected")
    ax.legend()
    plt.tight_layout()
    plt.savefig("skill_hire_rates.png", dpi=150)
    plt.show()
    print("Saved -> skill_hire_rates.png")

# USER INPUT

def encode_user_input(skills, experience, education, certification, job_role):
    """
    Convert user-provided resume details into a feature vector.
    Skills not in the training set are silently ignored (as planned).
    """
    row = {
        "Skills": skills,
        "Experience (Years)": experience,
        "Education": education,
        "Certifications": certification,
        "Job Role": job_role,
    }
    return np.array([encode_row(row)], dtype=float)

def predict_fit_score(model, scaler, skills, experience, education, certification, job_role):
    X = encode_user_input(skills, experience, education, certification, job_role)
    X_scaled = scaler.transform(X)
    prob_hire = model.predict_proba(X_scaled)[0][1] # P(Hire)
    return round(prob_hire * 100, 1)

def get_user_input():
    print("\n==============================================")
    print("\t\tAI RESUME FIT SCORE CALCULATOR")
    print("==============================================")

    # Job role
    print("\nAvailable job roles:")
    for i, role in enumerate(all_jobs, 1):
        print(f"\t{i}. {role}")
    role_idx = int(input("Select job role (1-4): ")) - 1
    job_role = all_jobs[role_idx]

    # Skills
    print(f"\nAvailable skills:\n\t{', '.join(all_skills)}")
    raw = input("Enter your skills (comma-separated): ")
    skills = [s.strip() for s in raw.split(",")]
    unknown = [s for s in skills if s not in all_skills]
    if unknown:
        print(f"\tUnknown skill(s) ignored: {unknown}")
    skills = [s for s in skills if s in all_skills]

    # Experience
    experience = int(input("Years of experience (0-10): "))

    # Education
    print(f"\nEducation levels: {', '.join(all_educations)}")
    education = input("Your education level: ").strip()

    # Certification
    print(f"\nCertifications: {', '.join(all_certs)}")
    certification = input("Your certification (or 'None'): ").strip()

    return skills, experience, education, certification, job_role

# MAIN

def main():
    # Load
    print("Loading dataset...")
    rows = load_data(path)
    print(f"\t{len(rows)} records loaded.")

    # Build feature matrix
    X, y = build_dataset(rows)
    print(f"\tFeature matrix shape: {X.shape} | Labels: {y.sum()} Hire / {(y == 0).sum()} Reject")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train
    print("\nTraining logistic regression model...")
    model, scaler = train_model(X_train, y_train)
    print("Done.")

    # Evaluate
    evaluate(model, scaler, X_test, y_test)

    # Visualise
    print("\nGenerating visualisations...")
    plot_hire_rate_by_role_and_edu(rows)
    plot_skill_hire_rates(rows)

    # Fit score
    while True:
        try:
            skills, experience, education, certification, job_role = get_user_input()
        except (ValueError, IndexError):
            print("Invalid input - please try again.")
            continue

        score = predict_fit_score(model, scaler, skills, experience, education, certification, job_role)

        print("\n------------------------------------------------")
        print(f"\tFit Score for {job_role}: {score}%")
        if score >= 75:
            print("Strong match - great fit for this role!")
        elif score >= 50:
            print("Moderate match - some gaps to address.")
        else:
            print("Low match - consider upskilling first.")
        print("------------------------------------------------")

        again = input("\nCheck another resume? (y/n): ").strip().lower()
        if again != "y":
            break
    print("\nGoodbye!")

if __name__ == "__main__":
    main()