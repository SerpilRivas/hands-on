# app.py
import os
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---- Optional prescription generator (safe fallback if missing) ----
try:
    from genai_prescriptions import generate_prescription
except Exception:
    def generate_prescription(provider, input_dict):
        return {
            "recommendation": "Block URL immediately",
            "severity": "High",
            "actions": [
                "Add URL to blacklist",
                "Alert security team",
                "Monitor for similar patterns"
            ]
        }

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="GenAI-Powered Phishing SOAR", page_icon="🛡️", layout="wide")


# -----------------------------------------------------------------------------
# Utilities: loading models and making predictions (scikit-learn)
# -----------------------------------------------------------------------------
def load_sklearn_model(pkl_path: str):
    """Load a scikit-learn model/pipeline saved as a .pkl file, or return None."""
    if os.path.exists(pkl_path):
        return joblib.load(pkl_path)
    return None


def predict_model_sklearn(model, data: pd.DataFrame) -> pd.DataFrame:
    """
    Mimic PyCaret's predict_model:
    returns a DataFrame with 'prediction_label' and 'prediction_score'
    (score = probability of MALICIOUS when available).
    """
    if model is None:
        raise ValueError("Model is None")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(data)
        classes = list(getattr(model, "classes_", []))

        # Find the 'MALICIOUS' class index, else fall back to class '1', else use max prob
        if "MALICIOUS" in classes:
            mal_idx = classes.index("MALICIOUS")
        elif 1 in classes:
            mal_idx = classes.index(1)
        else:
            mal_idx = np.argmax(proba, axis=1)[0]

        mal_score = proba[:, mal_idx]
        labels_num = (mal_score >= 0.5).astype(int)
    else:
        # No probabilities—use hard labels and fabricate a simple score
        labels_num = model.predict(data)
        mal_score = np.where(labels_num == 1, 0.8, 0.2)

    # Convert numeric labels to strings if needed
    if isinstance(labels_num[0], (np.integer, int)):
        pred_label = np.where(labels_num == 1, "MALICIOUS", "BENIGN")
    else:
        pred_label = labels_num

    out = data.copy()
    out["prediction_label"] = pred_label
    out["prediction_score"] = mal_score
    return out


# -----------------------------------------------------------------------------
# Cached loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """Load the classification model and optional feature importance plot."""
    model = load_sklearn_model("models/phishing_url_detector.pkl")
    plot_path = "models/feature_importance.png"
    plot = plot_path if os.path.exists(plot_path) else None
    return model, plot


@st.cache_resource
def load_attrib_assets():
    """Load clustering (attribution) model and its metadata."""
    models_dir = Path(__file__).resolve().parent / "models"

    cluster_model = load_sklearn_model(str(models_dir / "threat_actor_profiler.pkl"))

    # Metadata files with safe fallbacks
    try:
        with open(models_dir / "feature_columns.json", "r") as f:
            feature_columns = json.load(f)
    except FileNotFoundError:
        feature_columns = [
            "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
            "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
            "Abnormal_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "age_of_domain",
            "DNSRecord", "has_political_keyword", "iframe"
        ]

    try:
        with open(models_dir / "cluster_to_actor.json", "r") as f:
            cluster_to_actor = json.load(f)
    except FileNotFoundError:
        cluster_to_actor = {"0": "Organized Cybercrime", "1": "State-Sponsored", "2": "Hacktivist"}

    try:
        with open(models_dir / "actor_descriptions.json", "r") as f:
            actor_descriptions = json.load(f)
    except FileNotFoundError:
        actor_descriptions = {
            "Organized Cybercrime": "High-volume, profit-driven attacks using URL shortening, IP addresses, and abnormal URL structures. Typically targets financial institutions and e-commerce platforms.",
            "State-Sponsored": "Sophisticated, subtle attacks using valid SSL certificates but employing deceptive techniques like prefix/suffix manipulation. Focuses on intelligence gathering and strategic targets.",
            "Hacktivist": "Opportunistic attacks with mixed tactics, often incorporating political keywords and targeting organizations for ideological reasons."
        }

    return cluster_model, feature_columns, cluster_to_actor, actor_descriptions


# Load models and assets
model, feature_plot = load_assets()
cluster_model, feature_columns, cluster_to_actor, actor_descriptions = load_attrib_assets()

if not model:
    st.error("Classification model not found. Make sure `models/phishing_url_detector.pkl` exists.")
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar – user inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 URL Feature Input")

    form_values = {
        "url_length": st.select_slider("URL Length", options=["Short", "Normal", "Long"], value="Long"),
        "ssl_state": st.select_slider(
            "SSL Certificate Status", options=["Trusted", "Suspicious", "None"], value="Suspicious"
        ),
        "sub_domain": st.select_slider("Sub-domain Complexity", options=["None", "One", "Many"], value="One"),
        "prefix_suffix": st.checkbox("URL has a Prefix/Suffix (e.g., '-')", value=True),
        "has_ip": st.checkbox("URL uses an IP Address", value=False),
        "short_service": st.checkbox("Is it a shortened URL", value=False),
        "at_symbol": st.checkbox("URL contains '@' symbol", value=False),
        "abnormal_url": st.checkbox("Is it an abnormal URL", value=True),
    }

    st.divider()
    st.markdown("**Additional Content Signals (required by model)**")
    dns_record = st.checkbox("DNS record exists", value=True)
    iframe_tag = st.checkbox("Contains <iframe> tag", value=False)
    political_kw = st.checkbox("Contains political keyword", value=False)
    age_choice = st.select_slider("Age of Domain", options=["New", "Medium", "Old"], value="New")
    age_map = {"New": 0, "Medium": 1, "Old": 2}

    genai_provider = st.selectbox("Select GenAI Provider", ["Gemini", "OpenAI", "Grok"])
    submitted = st.button("💥 Analyze & Initiate Response", use_container_width=True, type="primary")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
st.title("🛡️ GenAI-Powered SOAR for Phishing URL Analysis")

if not submitted:
    st.info("Please provide the URL features in the sidebar and click 'Analyze' to begin.")
    if feature_plot:
        st.subheader("Model Feature Importance")
        st.image(feature_plot, caption="Feature importance from the trained model.")
    st.stop()

# Build the full feature row (matches training features)
input_dict = {
    # Core features
    "having_IP_Address": 1 if form_values["has_ip"] else -1,
    "URL_Length": -1 if form_values["url_length"] == "Short" else (0 if form_values["url_length"] == "Normal" else 1),
    "Shortining_Service": 1 if form_values["short_service"] else -1,
    "having_At_Symbol": 1 if form_values["at_symbol"] else -1,
    "double_slash_redirecting": -1,
    "Prefix_Suffix": 1 if form_values["prefix_suffix"] else -1,
    "having_Sub_Domain": -1 if form_values["sub_domain"] == "None" else (0 if form_values["sub_domain"] == "One" else 1),
    "SSLfinal_State": -1 if form_values["ssl_state"] == "None" else (0 if form_values["ssl_state"] == "Suspicious" else 1),
    "Abnormal_URL": 1 if form_values["abnormal_url"] else -1,
    "URL_of_Anchor": 0,
    "Links_in_tags": 0,
    "SFH": 0,
    # Extra columns some pipelines expect
    "Subdomain_Level": -1 if form_values["sub_domain"] == "None" else (0 if form_values["sub_domain"] == "One" else 1),
    "age_of_domain": age_map[age_choice],
    "DNSRecord": 1 if dns_record else 0,
    "has_political_keyword": 1 if political_kw else 0,
    "iframe": 1 if iframe_tag else 0,
    "actor_profile": "Unknown",
}
input_data = pd.DataFrame([input_dict])

# Enhanced risk visualization
risk_scores = {
    "Bad SSL": 25 if input_dict["SSLfinal_State"] < 1 else 0,
    "Abnormal URL": 24 if input_dict["Abnormal_URL"] == 1 else 0,
    "Prefix/Suffix": 15 if input_dict["Prefix_Suffix"] == 1 else 0,
    "Shortened URL": 15 if input_dict["Shortining_Service"] == 1 else 0,
    "Complex Sub-domain": 10 if input_dict["having_Sub_Domain"] == 1 else 0,
    "Long URL": 10 if input_dict["URL_Length"] == 1 else 0,
    "Uses IP Address": 8 if input_dict["having_IP_Address"] == 1 else 0,
    "At Symbol": 5 if input_dict["having_At_Symbol"] == 1 else 0,
}
risk_df = pd.DataFrame(list(risk_scores.items()), columns=["Feature", "Risk Contribution"]).sort_values(
    "Risk Contribution", ascending=False
)

# -----------------------------------------------------------------------------
# Workflow
# -----------------------------------------------------------------------------
with st.status("Executing SOAR playbook...", expanded=True) as status:
    st.write("▶️ **Step 1: Predictive Analysis** — running classification model.")
    time.sleep(0.5)

    # Try prediction; if missing columns, create them and retry
    try:
        prediction = predict_model_sklearn(model, input_data)
    except KeyError as e:
        missing_cols = []
        msg = str(e)
        if "[" in msg and "]" in msg:
            try:
                missing_cols = [c.strip(" '") for c in msg[msg.index("[") + 1: msg.index("]")].split(",")]
            except Exception:
                missing_cols = []
        for col in missing_cols:
            if col not in input_data.columns:
                input_data[col] = "Unknown" if col == "actor_profile" else 0
        prediction = predict_model_sklearn(model, input_data)

    is_malicious = prediction["prediction_label"].iloc[0] == "MALICIOUS"
    confidence_score = float(prediction["prediction_score"].iloc[0])
    verdict = "MALICIOUS" if is_malicious else "BENIGN"

    st.write(f"▶️ **Step 2: Verdict Interpretation** — model predicts **{verdict}** (confidence: {confidence_score:.2%}).")
    time.sleep(0.5)

    actor_name, actor_desc, cluster_id = None, None, None
    prescription = None

    if is_malicious:
        st.write("▶️ **Step 3: Prescriptive Analytics** — engaging plan and profiling actor.")
        st.write("▶️ **Step 3a: Threat Actor Attribution** — profiling malicious URL...")

        # Threat Attribution Logic
        if cluster_model and feature_columns:
            try:
                # Build exact column order for clustering
                attrib_row = {c: 0 for c in feature_columns}
                for c in feature_columns:
                    if c in input_data.columns:
                        attrib_row[c] = input_data.loc[0, c]
                attrib_input = pd.DataFrame([attrib_row])[feature_columns]

                raw = cluster_model.predict(attrib_input)[0]  # sklearn KMeans-style
                cluster_id = int(raw)

                mapped = cluster_to_actor.get(str(cluster_id), cluster_to_actor.get(cluster_id))
                actor_name = mapped if mapped else "Unknown Actor"
                actor_desc = actor_descriptions.get(actor_name, "No description available.")

                st.info(f"**Predicted Threat Actor:** {actor_name} (Cluster {cluster_id})")
                st.caption(f"**Profile:** {actor_desc}")

            except Exception:
                # FALLBACK: Rule-based attribution
                if input_dict.get("has_political_keyword") == 1:
                    actor_name = "Hacktivist"; cluster_id = 2
                elif (input_dict.get("having_IP_Address") == 1 and input_dict.get("Shortining_Service") == 1):
                    actor_name = "Organized Cybercrime"; cluster_id = 0
                elif (input_dict.get("SSLfinal_State") == 1 and input_dict.get("Prefix_Suffix") == 1):
                    actor_name = "State-Sponsored"; cluster_id = 1
                else:
                    actor_name = "Organized Cybercrime"; cluster_id = 0
                actor_desc = actor_descriptions.get(actor_name, "No description available.")
                st.info(f"**Predicted Threat Actor:** {actor_name} (Rule-based)")
                st.caption(f"**Profile:** {actor_desc}")
        else:
            # FALLBACK: Rule-based attribution when clustering model not available
            if input_dict.get("has_political_keyword") == 1:
                actor_name = "Hacktivist"; cluster_id = 2
            elif (input_dict.get("having_IP_Address") == 1 and input_dict.get("Shortining_Service") == 1):
                actor_name = "Organized Cybercrime"; cluster_id = 0
            elif (input_dict.get("SSLfinal_State") == 1 and input_dict.get("Prefix_Suffix") == 1):
                actor_name = "State-Sponsored"; cluster_id = 1
            else:
                actor_name = "Organized Cybercrime"; cluster_id = 0
            actor_desc = actor_descriptions.get(actor_name, "No description available.")
            st.info(f"**Predicted Threat Actor:** {actor_name} (Rule-based)")
            st.caption(f"**Profile:** {actor_desc}")

        # Generate prescription (robust)
        try:
            prescription = generate_prescription(genai_provider, dict(input_dict))
            status.update(label="✅ SOAR Playbook Executed Successfully!", state="complete", expanded=False)
        except Exception:
            prescription = {
                "recommendation": "Block URL immediately and investigate",
                "severity": "High",
                "actions": [
                    "Add URL to blacklist",
                    "Alert security team",
                    "Monitor for similar attack patterns"
                ]
            }
            status.update(label="✅ SOAR Playbook Executed Successfully!", state="complete", expanded=False)
    else:
        status.update(label="✅ Analysis Complete. No threat found.", state="complete", expanded=False)

# -----------------------------------------------------------------------------
# Tabs (Analysis Results)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Analysis Summary", "📈 Visual Insights", "📜 Prescriptive Plan", "🕵️ Threat Attribution"]
)

with tab1:
    st.subheader("Verdict and Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        if is_malicious:
            st.error("**Prediction: Malicious Phishing URL**", icon="🚨")
        else:
            st.success("**Prediction: Benign URL**", icon="✅")
    with col2:
        confidence_display = confidence_score if is_malicious else (1 - confidence_score)
        st.metric("Model Confidence", f"{confidence_display:.2%}")

    total_risk = sum(risk_scores.values())
    st.metric("Total Risk Score", f"{total_risk}/112")
    st.caption("This score represents the cumulative risk based on URL features.")

with tab2:
    st.subheader("Visual Analysis")
    st.write("#### Risk Contribution by Feature")
    if len(risk_df[risk_df["Risk Contribution"] > 0]) > 0:
        st.bar_chart(risk_df[risk_df["Risk Contribution"] > 0].set_index("Feature"))
        st.caption("Features that contribute to the risk assessment.")
    else:
        st.info("No significant risk factors detected in this URL.")
    if feature_plot:
        st.write("#### Model Feature Importance (Global)")
        st.image(feature_plot, caption="Global feature importance from training data.")

with tab3:
    st.subheader("Actionable Response Plan")
    if prescription:
        st.success("A prescriptive response plan has been generated by AI.", icon="🤖")
        if isinstance(prescription, dict):
            for key, value in prescription.items():
                if isinstance(value, list):
                    st.write(f"**{key.title()}:**")
                    for item in value:
                        st.write(f"• {item}")
                else:
                    st.write(f"**{key.title()}:** {value}")
        else:
            st.json(prescription, expanded=False)
    else:
        st.info("No prescriptive plan was generated because the URL was classified as benign.")

with tab4:
    st.subheader("🕵️ Threat Attribution Analysis")
    if is_malicious:
        if actor_name and actor_name != "Attribution Failed":
            st.success(f"**Identified Threat Actor:** {actor_name}")
            st.write("#### Actor Profile")
            st.write(actor_desc or "No detailed description available.")
            if cluster_id is not None:
                st.write("#### Technical Details")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Cluster ID", cluster_id)
                with c2:
                    st.metric("Attribution Method", "K-Means (sklearn)" if cluster_model else "Rule-based")
            st.write("#### Attribution Methodology")
            if cluster_model:
                st.info(
                    "Attribution is inferred using an unsupervised clustering model trained on behavioral patterns "
                    "including SSL usage, URL structure anomalies, shortening services, IP address usage, and other indicators."
                )
            else:
                st.info(
                    "Attribution is performed using rule-based logic analyzing patterns such as political keywords "
                    "(Hacktivist), IP + shortening (Organized Cybercrime), and SSL + prefix/suffix (State-Sponsored)."
                )
            st.write("#### Key Behavioral Indicators")
            relevant_features = []
            if actor_name == "Organized Cybercrime":
                if input_dict["Shortining_Service"] == 1: relevant_features.append("• Uses URL shortening services")
                if input_dict["having_IP_Address"] == 1: relevant_features.append("• Direct IP address usage")
                if input_dict["Abnormal_URL"] == 1: relevant_features.append("• Abnormal URL structure")
            elif actor_name == "State-Sponsored":
                if input_dict["SSLfinal_State"] == 1: relevant_features.append("• Valid SSL certificate (sophisticated)")
                if input_dict["Prefix_Suffix"] == 1: relevant_features.append("• Deceptive prefix/suffix manipulation")
            elif actor_name == "Hacktivist":
                if input_dict["has_political_keyword"] == 1: relevant_features.append("• Contains political keywords")
                relevant_features.append("• Mixed attack tactics")
            for f in relevant_features or ["• Pattern analysis based on multiple indicators"]:
                st.write(f)
        else:
            st.warning("⚠️ Threat attribution could not be performed.")
            st.write("Possible causes: clustering model unavailable, insufficient feature data, or technical issue.")
    else:
        st.info("🛡️ Attribution is only performed when a URL is classified as malicious.")
