"""
preprocess_data.py
------------------
Reads raw Kaggle CSVs, cleans text, and produces structured instruction-tuning
JSON examples in  data/processed/instructions.json

Output schema per example:
{
    "instruction": "...",   # task prompt
    "input":       "...",   # resume snippet
    "output":      "..."    # desired model response
}

Usage:
  python pipelines/preprocess_data.py
"""

import json
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

RESUME_CSV = RAW_DIR / "UpdatedResumeDataSet.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── Instruction templates ────────────────────────────────────────────────────

INSTRUCTION_TEMPLATES = [
    "Analyze this resume and suggest specific improvements to make it stronger.",
    "Review the following resume and identify missing skills or experiences.",
    "Given this resume, what are the top 3 weaknesses and how can they be fixed?",
    "Evaluate this resume and provide actionable feedback for a {category} role.",
    "Identify skill gaps in this resume for someone applying to {category} jobs.",
    "Rewrite the summary section of this resume to be more impactful.",
    "List 5 interview questions a recruiter might ask based on this resume.",
    "What quantifiable achievements is this resume missing?",
    "Score this resume out of 10 and justify your rating.",
    "How can this resume be tailored for a senior {category} position?",
]

OUTPUT_TEMPLATES = [
    (
        "Based on the resume provided, here are the key improvements:\n\n"
        "1. **Add Quantifiable Achievements**: Replace vague statements with "
        "measurable results (e.g., 'Increased system performance by 30%').\n"
        "2. **Strengthen the Summary**: The opening statement lacks impact. "
        "Lead with your unique value proposition.\n"
        "3. **Include Relevant Keywords**: Add industry-specific keywords aligned "
        "with {category} job descriptions to pass ATS filters.\n"
        "4. **Show Career Progression**: Clearly demonstrate growth in responsibilities.\n"
        "5. **Certifications**: Consider adding relevant certifications to stand out."
    ),
    (
        "This resume has a solid foundation for a {category} role, but is missing:\n\n"
        "• **Metrics and impact numbers** — quantify your contributions.\n"
        "• **Technical stack specifics** — list exact tools/frameworks used.\n"
        "• **Leadership examples** — even informal mentoring is valuable.\n"
        "• **Projects section** — add 2-3 notable projects with outcomes.\n\n"
        "Overall Score: 6/10. With these additions it could reach 8.5/10."
    ),
    (
        "Interview questions a recruiter might ask:\n\n"
        "1. Walk me through your most challenging {category} project.\n"
        "2. How do you keep up with fast-changing technologies in this field?\n"
        "3. Describe a situation where you had to make a technical decision under pressure.\n"
        "4. What is your approach to debugging complex production issues?\n"
        "5. How have you collaborated with cross-functional teams on large deliverables?\n\n"
        "Prepare STAR-format answers (Situation, Task, Action, Result) for best results."
    ),
]


# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize, strip URLs, extra whitespace from resume text."""
    if not isinstance(text, str):
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)
    # Remove phone numbers
    text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", "[PHONE]", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate(text: str, max_chars: int = 800) -> str:
    """Truncate to max_chars, ending at a word boundary."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


# ── Dataset builders ─────────────────────────────────────────────────────────

def build_resume_examples(df: pd.DataFrame) -> list[dict]:
    examples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building resume examples"):
        resume_text = clean_text(str(row.get("Resume", "")))
        category = str(row.get("Category", "technology")).strip()

        if len(resume_text) < 100:
            continue  # skip near-empty rows

        resume_snippet = truncate(resume_text)
        tmpl = random.choice(INSTRUCTION_TEMPLATES)
        instruction = tmpl.format(category=category)
        output_tmpl = random.choice(OUTPUT_TEMPLATES)
        output = output_tmpl.format(category=category)

        examples.append({
            "instruction": instruction,
            "input": resume_snippet,
            "output": output,
        })

    return examples


def build_filler_examples(categories: list[str], n: int = 500) -> list[dict]:
    """
    Generate synthetic skill-gap examples when resume dataset is small,
    ensuring we always have a rich training set.
    """
    skill_map = {
        "Data Science":   ["Python", "SQL", "Machine Learning", "Statistics", "Tableau"],
        "Engineering":    ["System Design", "Microservices", "CI/CD", "Docker", "Kubernetes"],
        "Finance":        ["Financial Modeling", "Excel VBA", "Bloomberg", "Risk Analysis"],
        "Marketing":      ["SEO", "Google Ads", "Content Strategy", "A/B Testing"],
        "HR":             ["Talent Acquisition", "HRIS", "Compensation Planning"],
        "Healthcare":     ["EHR Systems", "Patient Care", "HIPAA Compliance"],
    }
    default_skills = ["Communication", "Leadership", "Problem Solving", "Agile"]
    examples = []

    for _ in range(n):
        cat = random.choice(categories) if categories else "technology"
        skills = skill_map.get(cat, default_skills)
        missing = random.sample(skills, min(2, len(skills)))
        resume_stub = (
            f"Experienced {cat} professional with {random.randint(2, 8)} years "
            f"of experience. Skilled in {', '.join(random.sample(skills, min(3, len(skills))))}."
        )
        output = (
            f"This resume is missing: {', '.join(missing)}.\n\n"
            "Recommendations:\n"
            f"1. Add a dedicated Skills section highlighting {missing[0]}.\n"
            f"2. Include a project that demonstrates {missing[-1]} expertise.\n"
            "3. Quantify past results (e.g., 'Reduced processing time by 40%').\n"
            "4. Tailor the summary to the target role specifically."
        )
        examples.append({
            "instruction": f"Identify skill gaps in this resume for a {cat} position.",
            "input": resume_stub,
            "output": output,
        })
    return examples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    all_examples: list[dict] = []

    # ── Resume dataset
    if RESUME_CSV.exists():
        print(f"📄  Loading {RESUME_CSV.name} ...")
        df = pd.read_csv(RESUME_CSV)
        print(f"   Shape: {df.shape}  |  Columns: {list(df.columns)}")
        resume_examples = build_resume_examples(df)
        categories = df["Category"].dropna().unique().tolist() if "Category" in df.columns else []
        print(f"   Generated {len(resume_examples)} examples from resumes.")
        all_examples.extend(resume_examples)
    else:
        print(f"⚠️   {RESUME_CSV} not found. Run kaggle_download.py first.")
        categories = ["Data Science", "Engineering", "Finance", "Marketing"]

    # ── Filler synthetic examples to reach target dataset size (≥ 3000)
    current = len(all_examples)
    target = 3000
    if current < target:
        filler_n = target - current
        print(f"\n🧪  Adding {filler_n} synthetic examples to reach {target} total ...")
        all_examples.extend(build_filler_examples(categories, n=filler_n))

    random.shuffle(all_examples)

    out_path = PROC_DIR / "instructions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Saved {len(all_examples)} examples → {out_path}")
    print("    Sample:")
    sample = random.choice(all_examples)
    for k, v in sample.items():
        print(f"    [{k}] {str(v)[:120]}")


if __name__ == "__main__":
    main()
