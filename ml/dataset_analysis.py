import pandas as pd


# ============================================================
# 1. LOAD ORIGINAL DATASET
# ============================================================

DATA_PATH = "data/support_tickets.csv"

df = pd.read_csv(DATA_PATH)

print("\nORIGINAL DATASET")
print("----------------")
print(f"Rows: {len(df)}")


# ============================================================
# 2. COMBINE TEXT
# ============================================================

df["full_text"] = (
    df["Ticket Type"].astype(str)
    + " "
    + df["Ticket Subject"].astype(str)
    + " "
    + df["Ticket Description"].astype(str)
).str.lower()


# ============================================================
# 3. CALCULATE MEANINGFUL SIGNALS
# ============================================================

urgent_words = [
    "urgent",
    "immediately",
    "as soon as possible",
    "emergency",
]

severe_words = [
    "not working",
    "stopped working",
    "broken",
    "failed",
    "crashing",
    "data loss",
    "security",
]

impact_words = [
    "affecting my work",
    "affecting my business",
    "cannot access",
    "unable to access",
    "multiple devices",
    "money",
    "payment",
]

normal_problem_words = [
    "problem",
    "issue",
    "error",
    "refund",
    "replacement",
    "complaint",
]


def contains_any(text, words):
    return any(word in text for word in words)


# ============================================================
# 4. CREATE FEATURES
# ============================================================

df["urgent_signal"] = df["full_text"].apply(
    lambda x: int(contains_any(x, urgent_words))
)

df["severe_signal"] = df["full_text"].apply(
    lambda x: int(contains_any(x, severe_words))
)

df["impact_signal"] = df["full_text"].apply(
    lambda x: int(contains_any(x, impact_words))
)

df["problem_signal"] = df["full_text"].apply(
    lambda x: int(contains_any(x, normal_problem_words))
)


# ============================================================
# 5. DERIVE PRIORITY
# ============================================================

def assign_priority(row):

    urgent = row["urgent_signal"]
    severe = row["severe_signal"]
    impact = row["impact_signal"]
    problem = row["problem_signal"]

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if (
        urgent
        and (severe or impact)
    ):
        return "Critical"

    if (
        severe
        and impact
        and problem
    ):
        return "Critical"

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if (
        severe
        and impact
    ):
        return "High"

    if (
        urgent
        and severe
    ):
        return "High"

    # --------------------------------------------------------
    # MEDIUM
    # --------------------------------------------------------

    if severe:
        return "Medium"

    if impact:
        return "Medium"

    if problem:
        return "Medium"

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    return "Low"


df["ML Priority"] = df.apply(
    assign_priority,
    axis=1
)


# ============================================================
# 6. CHECK DISTRIBUTION
# ============================================================

print("\nNEW ML PRIORITY DISTRIBUTION")
print("----------------------------")

print(
    df["ML Priority"].value_counts()
)


print("\nPERCENTAGE DISTRIBUTION")
print("-----------------------")

print(
    (df["ML Priority"].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 7. SHOW EXAMPLES
# ============================================================

print("\nEXAMPLES")
print("--------")

print(
    df[
        [
            "Ticket Type",
            "Ticket Subject",
            "urgent_signal",
            "severe_signal",
            "impact_signal",
            "problem_signal",
            "ML Priority",
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 8. SAVE ML DATASET
# ============================================================

output_columns = [
    "Ticket ID",
    "Ticket Type",
    "Ticket Subject",
    "Ticket Description",
    "Customer Age",
    "Customer Gender",
    "Product Purchased",
    "Ticket Channel",
    "Ticket Status",
    "urgent_signal",
    "severe_signal",
    "impact_signal",
    "problem_signal",
    "ML Priority",
]


df[output_columns].to_csv(
    "data/priority_training.csv",
    index=False
)


print("\nDATASET SAVED")
print("-------------")
print("data/priority_training.csv")
print(f"Rows: {len(df)}")