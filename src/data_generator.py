import os
import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic phone call metadata dataset for training spam/fraud classifiers.
    
    Features:
    - calls_per_hour: Frequency of outgoing calls from caller node.
    - avg_call_duration: Average duration of call in seconds (short/dropped for spam).
    - contact_degree: Number of common contacts shared with the recipient.
    - spam_report_count: Crowdsourced spam flags in the last 24 hours.
    - is_spam: Target label (1 = spam/fraud, 0 = legitimate).
    """
    np.random.seed(random_state)
    
    num_legit = int(num_samples * 0.70)
    num_spam = num_samples - num_legit
    
    # ---------------------------------------------------------
    # Legitimate Caller Distribution (70% of dataset)
    # ---------------------------------------------------------
    # calls_per_hour: low (1 to 8 calls/hr)
    legit_calls_per_hour = np.random.gamma(shape=2.0, scale=1.5, size=num_legit) + 0.5
    legit_calls_per_hour = np.clip(legit_calls_per_hour, 0.1, 15.0)
    
    # avg_call_duration: higher/normal conversations (30s to 600s)
    legit_avg_duration = np.random.normal(loc=180.0, scale=60.0, size=num_legit)
    legit_avg_duration = np.clip(legit_avg_duration, 25.0, 1200.0)
    
    # contact_degree: moderate to high shared contacts (2 to 20)
    legit_contact_degree = np.random.poisson(lam=6.0, size=num_legit)
    legit_contact_degree = np.clip(legit_contact_degree, 1, 30)
    
    # spam_report_count: very low or zero
    legit_spam_reports = np.random.poisson(lam=0.1, size=num_legit)
    legit_spam_reports = np.clip(legit_spam_reports, 0, 2)

    # ---------------------------------------------------------
    # Spam / Robocaller / Fraud Distribution (30% of dataset)
    # ---------------------------------------------------------
    # calls_per_hour: high volume robocalls (20 to 120 calls/hr)
    spam_calls_per_hour = np.random.gamma(shape=5.0, scale=8.0, size=num_spam) + 15.0
    spam_calls_per_hour = np.clip(spam_calls_per_hour, 12.0, 150.0)
    
    # avg_call_duration: short, hung up, or automated voicemails (1s to 25s)
    spam_avg_duration = np.random.exponential(scale=8.0, size=num_spam) + 1.0
    spam_avg_duration = np.clip(spam_avg_duration, 0.5, 45.0)
    
    # contact_degree: zero or rare shared contacts (0 to 1)
    spam_contact_degree = np.random.binomial(n=1, p=0.08, size=num_spam)
    
    # spam_report_count: high crowdsourced report counts (5 to 50+)
    spam_spam_reports = np.random.gamma(shape=3.0, scale=5.0, size=num_spam) + 3.0
    spam_spam_reports = np.clip(spam_spam_reports, 1, 100).astype(int)

    # Combine into dataframes
    legit_df = pd.DataFrame({
        "calls_per_hour": np.round(legit_calls_per_hour, 2),
        "avg_call_duration": np.round(legit_avg_duration, 2),
        "contact_degree": legit_contact_degree,
        "spam_report_count": legit_spam_reports,
        "is_spam": 0
    })

    spam_df = pd.DataFrame({
        "calls_per_hour": np.round(spam_calls_per_hour, 2),
        "avg_call_duration": np.round(spam_avg_duration, 2),
        "contact_degree": spam_contact_degree,
        "spam_report_count": spam_spam_reports,
        "is_spam": 1
    })

    df = pd.concat([legit_df, spam_df], ignore_index=True)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

def save_synthetic_dataset(output_path: str = "data/synthetic_calls.csv", num_samples: int = 5000) -> str:
    """Generates dataset and saves to specified CSV file path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_synthetic_data(num_samples=num_samples)
    df.to_csv(output_path, index=False)
    print(f"[+] Dataset saved to '{output_path}' ({len(df)} records).")
    return output_path

if __name__ == "__main__":
    save_synthetic_dataset()
