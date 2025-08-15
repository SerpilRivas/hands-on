# train_model.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycaret.classification import ClassificationExperiment
from pycaret.clustering import ClusteringExperiment

RNG_SEED = 42
np.random.seed(RNG_SEED)

# Save artifacts under ./models (already exists in your project)
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Synthetic data with 3 malicious actor profiles + benign baseline
# -------------------------------------------------------------------
def bernoulli(p: float, n: int) -> np.ndarray:
    return (np.random.rand(n) < p).astype(int)

def normal_clip(mu: float, sigma: float, n: int, lo: float, hi: float) -> np.ndarray:
    x = np.random.normal(mu, sigma, n)
    return np.clip(x, lo, hi)

def make_profile(n: int, profile: str) -> pd.DataFrame:
    """
    Create n rows of MALICIOUS data for a specific actor profile.
    Features intentionally echo common URL heuristics so clustering can 'discover' groups.
    """
    df = pd.DataFrame(index=range(n))

    if profile == "State-Sponsored":
        # Well-resourced & subtle: valid SSL, some hyphen tricks, almost never IP/shorteners,
        # older domains with proper DNS; moderate URL complexity.
        df["SSLfinal_State"]      = bernoulli(0.85, n)
        df["Prefix_Suffix"]       = bernoulli(0.60, n)
        df["Shortining_Service"]  = bernoulli(0.10, n)
        df["having_IP_Address"]   = bernoulli(0.05, n)
        df["URL_Length"]          = np.random.choice([0,1,2], size=n, p=[0.25,0.55,0.20])
        df["Subdomain_Level"]     = np.random.choice([0,1,2], size=n, p=[0.20,0.55,0.25])
        df["age_of_domain"]       = normal_clip(36, 10, n, 3, 120)
        df["DNSRecord"]           = bernoulli(0.95, n)
        df["has_political_keyword"] = bernoulli(0.20, n)
        df["Iframe"]              = bernoulli(0.15, n)
        df["URL_of_Anchor"]       = bernoulli(0.25, n)

    elif profile == "Organized Cybercrime":
        # Noisy & high volume: IPs, shorteners, hyphens, longer URLs, many subdomains,
        # young disposable domains, frequent iframes/odd anchors.
        df["SSLfinal_State"]      = bernoulli(0.35, n)
        df["Prefix_Suffix"]       = bernoulli(0.70, n)
        df["Shortining_Service"]  = bernoulli(0.70, n)
        df["having_IP_Address"]   = bernoulli(0.60, n)
        df["URL_Length"]          = np.random.choice([0,1,2], size=n, p=[0.10,0.35,0.55])
        df["Subdomain_Level"]     = np.random.choice([0,1,2], size=n, p=[0.15,0.35,0.50])
        df["age_of_domain"]       = normal_clip(6, 4, n, 1, 24)
        df["DNSRecord"]           = bernoulli(0.80, n)
        df["has_political_keyword"] = bernoulli(0.05, n)
        df["Iframe"]              = bernoulli(0.35, n)
        df["URL_of_Anchor"]       = bernoulli(0.50, n)

    elif profile == "Hacktivist":
        # Opportunistic & cause-driven: political keywords common, mixed tradecraft,
        # medium age, DNS sometimes flaky.
        df["SSLfinal_State"]      = bernoulli(0.45, n)
        df["Prefix_Suffix"]       = bernoulli(0.45, n)
        df["Shortining_Service"]  = bernoulli(0.40, n)
        df["having_IP_Address"]   = bernoulli(0.20, n)
        df["URL_Length"]          = np.random.choice([0,1,2], size=n, p=[0.30,0.45,0.25])
        df["Subdomain_Level"]     = np.random.choice([0,1,2], size=n, p=[0.35,0.45,0.20])
        df["age_of_domain"]       = normal_clip(12, 6, n, 1, 36)
        df["DNSRecord"]           = bernoulli(0.65, n)
        df["has_political_keyword"] = bernoulli(0.70, n)
        df["Iframe"]              = bernoulli(0.25, n)
        df["URL_of_Anchor"]       = bernoulli(0.35, n)

    else:
        raise ValueError("Unknown profile")

    df["actor_profile"] = profile
    df["label"] = "MALICIOUS"
    return df

def make_benign(n: int) -> pd.DataFrame:
    df = pd.DataFrame(index=range(n))
    df["SSLfinal_State"]      = bernoulli(0.90, n)
    df["Prefix_Suffix"]       = bernoulli(0.05, n)
    df["Shortining_Service"]  = bernoulli(0.05, n)
    df["having_IP_Address"]   = bernoulli(0.01, n)
    df["URL_Length"]          = np.random.choice([0,1,2], size=n, p=[0.55,0.40,0.05])
    df["Subdomain_Level"]     = np.random.choice([0,1,2], size=n, p=[0.60,0.35,0.05])
    df["age_of_domain"]       = normal_clip(48, 12, n, 6, 200)
    df["DNSRecord"]           = bernoulli(0.98, n)
    df["has_political_keyword"] = bernoulli(0.01, n)
    df["Iframe"]              = bernoulli(0.02, n)
    df["URL_of_Anchor"]       = bernoulli(0.03, n)
    df["actor_profile"] = "Benign"
    df["label"] = "BENIGN"
    return df

def generate_synthetic_data(total_rows: int = 6000, malicious_ratio: float = 0.5) -> pd.DataFrame:
    m = int(total_rows * malicious_ratio)
    b = total_rows - m
    # split malicious across 3 profiles
    base = m // 3
    sizes = [base, base, m - 2*base]

    state = make_profile(sizes[0], "State-Sponsored")
    crime = make_profile(sizes[1], "Organized Cybercrime")
    hackt = make_profile(sizes[2], "Hacktivist")
    benign = make_benign(b)

    df = pd.concat([state, crime, hackt, benign], ignore_index=True)
    df = df.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)
    return df

# -------------------------------------------------------------------
# Train classifier (binary) and save feature order
# -------------------------------------------------------------------
def train_classifier(df: pd.DataFrame) -> list[str]:
    exp = ClassificationExperiment()
    exp.setup(
        data=df,
        target="label",
        session_id=RNG_SEED,
        normalize=True,
        imputation_type="simple",
        fold=5,
        verbose=False,
    )
    best = exp.compare_models(sort="F1")
    final = exp.finalize_model(best)
    exp.save_model(final, str(MODELS_DIR / "phishing_url_detector"))

    feature_cols = [c for c in df.columns if c not in ("label", "actor_profile")]
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    return feature_cols

# -------------------------------------------------------------------
# Train clustering (3 clusters) on malicious-only subset
# -------------------------------------------------------------------
def _majority_map(cluster_series: pd.Series, truth_series: pd.Series) -> dict[int, str]:
    mapping = {}
    for k in sorted(cluster_series.unique()):
        # If k is like 'Cluster 0', extract the number
        if isinstance(k, str) and k.startswith("Cluster "):
            k_num = int(k.replace("Cluster ", "").strip())
        else:
            k_num = int(k)
        mask = cluster_series == k
        top = truth_series[mask].value_counts().idxmax()
        mapping[k_num] = str(top)
    return mapping

def train_clusterer(df: pd.DataFrame, feature_cols: list[str]) -> None:
    mal = df[df["label"] == "MALICIOUS"].copy()
    X = mal[feature_cols].copy()

    cexp = ClusteringExperiment()
    cexp.setup(data=X, session_id=RNG_SEED, normalize=True, verbose=False)
    model = cexp.create_model("kmeans", num_clusters=3)
    assigned = cexp.assign_model(model)
    clusters = assigned["Cluster"].reset_index(drop=True)
    truth = mal["actor_profile"].reset_index(drop=True)
    mapping = _majority_map(clusters, truth)

    cexp.save_model(model, str(MODELS_DIR / "threat_actor_profiler"))
    (MODELS_DIR / "cluster_to_actor.json").write_text(json.dumps(mapping, indent=2))

    descriptions = {
        "State-Sponsored": "Well-resourced; valid SSL and subtle domain tricks; long-term objectives; strong OPSEC.",
        "Organized Cybercrime": "Financially motivated; high-volume noisy campaigns; shorteners/IPs; disposable infra.",
        "Hacktivist": "Cause-driven; opportunistic; political keywords; variable tradecraft; shorter-lived infra."
    }
    (MODELS_DIR / "actor_descriptions.json").write_text(json.dumps(descriptions, indent=2))

# -------------------------------------------------------------------
def train():
    print(">> Generating synthetic data…")
    df = generate_synthetic_data(total_rows=6000, malicious_ratio=0.5)
    print(df.head())

    print(">> Training classifier…")
    feature_cols = train_classifier(df)

    print(">> Training 3-cluster profiler on malicious subset…")
    train_clusterer(df, feature_cols)

    print("\n✔ Artifacts saved in:", MODELS_DIR.resolve())
    print("   - phishing_url_detector.pkl")
    print("   - threat_actor_profiler.pkl")
    print("   - feature_columns.json")
    print("   - cluster_to_actor.json")
    print("   - actor_descriptions.json")

if __name__ == "__main__":
    train()
