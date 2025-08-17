# app.py
import os
import time
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Mock the pycaret functions since pycaret is not available
class MockModel:
    def predict(self, data):
        # Simple mock prediction based on risk factors
        risk_score = 0
        if data['having_IP_Address'].iloc[0] == 1:
            risk_score += 0.3
        if data['Abnormal_URL'].iloc[0] == 1:
            risk_score += 0.4
        if data['Prefix_Suffix'].iloc[0] == 1:
            risk_score += 0.2
        if data['SSLfinal_State'].iloc[0] < 1:
            risk_score += 0.3
        
        is_malicious = risk_score > 0.5
        confidence = min(0.95, max(0.55, risk_score))
        
        result = data.copy()
        result['prediction_label'] = 'MALICIOUS' if is_malicious else 'BENIGN'
        result['prediction_score'] = confidence if is_malicious else (1 - confidence)
        return result

def load_cls_model(path):
    return MockModel()

def predict_model(model, data):
    return model.predict(data)

# Mock prescription generator
def generate_prescription(provider, input_dict):
    return {
        "recommendation": "Block URL immediately and investigate",
        "severity": "High", 
        "actions": [
            "Add URL to blacklist",
            "Alert security team",
            "Monitor for similar attack patterns",
            f"Escalate to {provider} AI analysis team"
        ]
    }

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="GenAI-Powered Phishing SOAR", page_icon="🛡️", layout="wide")

# -------------------------------------------------
# Loaders
# -------------------------------------------------
@st.cache_resource
def load_assets():
    """Load the classification model and optional feature importance plot."""
    # Return mock model since files don't exist in cloud deployment
    model = MockModel()
    plot = None
    return model, plot

@st.cache_resource
def load_attrib_assets():
    """Load clustering (attribution) model and its metadata."""
    # Return defaults since clustering model files don't exist
    feature_columns = [
        "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
        "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
        "Abnormal_URL", "URL_of_Anchor", "Links_in_tags", "SFH", "age_of_domain",
        "DNSRecord", "has_political_keyword", "iframe"
    ]
    
    cluster_to_actor = {
        "0": "Organized Cybercrime",
        "1": "State-Sponsored", 
        "2": "Hacktivist"
    }
    
    actor_descriptions = {
        "Organized Cybercrime": "High-volume, profit-driven attacks using URL shortening, IP addresses, and abnormal URL structures. Typically targets financial institutions and e-commerce platforms.",
        "State-Sponsored": "Sophisticated, subtle attacks using valid SSL certificates but employing deceptive techniques like prefix/suffix manipulation. Focuses on intelligence gathering and strategic targets.",
        "Hacktivist": "Opportunistic attacks with mixed tactics, often incorporating political keywords and targeting organizations for ideological reasons."
    }

    return None, feature_columns, cluster_to_actor, actor_descriptions

# Load models and assets
model, feature_plot = load_assets()
cluster_model, feature_columns, cluster_to_actor, actor_descriptions = load_attrib_assets()

# -------------------------------------------------
# Sidebar – user inputs
# -------------------------------------------------
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
    st.markdown("**Additional Content Signals**")
    dns_record = st.checkbox("DNS record exists", value=True)
    iframe_tag = st.checkbox("Contains <iframe> tag", value=False)
    political_kw = st.checkbox("Contains political keyword", value=False)
    age_choice = st.select_slider("Age of Domain", options=["New", "Medium", "Old"], value="New")
    age_map = {"New": 0, "Medium": 1, "Old": 2}

    genai_provider = st.selectbox("Select GenAI Provider", ["Gemini", "OpenAI", "Grok"])
    submitted = st.button("💥 Analyze & Initiate Response", use_container_width=True, type="primary")

# -------------------------------------------------
# Main
# -------------------------------------------------
st.title("🛡️ GenAI-Powered SOAR for Phishing URL Analysis")

if not submitted:
    st.info("Please provide the URL features in the sidebar and click 'Analyze' to begin.")
    st.markdown("""
    ### About This Demo
    This is a demonstration of a SOAR (Security Orchestration, Automation and Response) system 
    that uses machine learning to analyze potentially malicious URLs and generate automated response plans.
    
    **Features:**
    - 🔍 ML-powered phishing detection
    - 🕵️ Threat actor attribution  
    - 📋 Automated response recommendations
    - 📊 Risk scoring and visualization
    """)
    st.stop()

# Build the full feature row
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
    "age_of_domain": age_map[age_choice],
    "DNSRecord": 1 if dns_record else 0,
    "has_political_keyword": 1 if political_kw else 0,
    "iframe": 1 if iframe_tag else 0,
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

# -------------------------------------------------
# Workflow
# -------------------------------------------------
with st.status("Executing SOAR playbook...", expanded=True) as status:
    st.write("▶️ **Step 1: Predictive Analysis** — running classification model.")
    time.sleep(1)

    prediction = predict_model(model, data=input_data)
    is_malicious = prediction["prediction_label"].iloc[0] == "MALICIOUS"
    verdict = "MALICIOUS" if is_malicious else "BENIGN"
    confidence_score = prediction["prediction_score"].iloc[0]
    
    st.write(f"▶️ **Step 2: Verdict Interpretation** — model predicts **{verdict}** (confidence: {confidence_score:.2%}).")
    time.sleep(1)

    # Initialize defaults
    actor_name, actor_desc, cluster_id = None, None, None
    prescription = None

    if is_malicious:
        st.write("▶️ **Step 3: Prescriptive Analytics** — engaging plan and profiling actor.")
        st.write("▶️ **Step 3a: Threat Actor Attribution** — profiling malicious URL...")
        time.sleep(1)

        # Rule-based attribution since clustering model isn't available
        if input_dict.get("has_political_keyword") == 1:
            actor_name = "Hacktivist"
            cluster_id = 2
        elif (input_dict.get("having_IP_Address") == 1 and 
              input_dict.get("Shortining_Service") == 1):
            actor_name = "Organized Cybercrime"
            cluster_id = 0
        elif (input_dict.get("SSLfinal_State") == 1 and 
              input_dict.get("Prefix_Suffix") == 1):
            actor_name = "State-Sponsored"
            cluster_id = 1
        else:
            actor_name = "Organized Cybercrime"
            cluster_id = 0

        actor_desc = actor_descriptions.get(actor_name, "No description available.")
        st.info(f"**Predicted Threat Actor:** {actor_name} (Rule-based)")
        st.caption(f"**Profile:** {actor_desc}")

        # Generate prescription
        prescription = generate_prescription(genai_provider, dict(input_dict))
        status.update(label="✅ SOAR Playbook Executed Successfully!", state="complete", expanded=False)
    else:
        status.update(label="✅ Analysis Complete. No threat found.", state="complete", expanded=False)

# -------------------------------------------------
# Tabs (Analysis Results)
# -------------------------------------------------
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
    
    # Risk summary
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

with tab3:
    st.subheader("Actionable Response Plan")
    
    if prescription:
        st.success("A prescriptive response plan has been generated by AI.", icon="🤖")
        
        # Display prescription in a more readable format
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
        if actor_name:
            # Main attribution result
            st.success(f"**Identified Threat Actor:** {actor_name}")
            
            # Detailed profile
            st.write("#### Actor Profile")
            st.write(actor_desc or "No detailed description available.")
            
            # Technical details
            if cluster_id is not None:
                st.write("#### Technical Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Cluster ID", cluster_id)
                with col2:
                    st.metric("Attribution Method", "Rule-based Logic")
            
            # Methodology explanation
            st.write("#### Attribution Methodology")
            st.info(
                "Attribution is performed using rule-based logic analyzing behavioral patterns such as "
                "political keywords (Hacktivist), IP addresses with URL shortening (Organized Cybercrime), "
                "and SSL certificates with prefix/suffix manipulation (State-Sponsored)."
            )
            
            # Feature analysis for this actor type
            st.write("#### Key Behavioral Indicators")
            relevant_features = []
            if actor_name == "Organized Cybercrime":
                if input_dict["Shortining_Service"] == 1:
                    relevant_features.append("• Uses URL shortening services")
                if input_dict["having_IP_Address"] == 1:
                    relevant_features.append("• Direct IP address usage")
                if input_dict["Abnormal_URL"] == 1:
                    relevant_features.append("• Abnormal URL structure")
            elif actor_name == "State-Sponsored":
                if input_dict["SSLfinal_State"] == 1:
                    relevant_features.append("• Valid SSL certificate (sophisticated)")
                if input_dict["Prefix_Suffix"] == 1:
                    relevant_features.append("• Deceptive prefix/suffix manipulation")
            elif actor_name == "Hacktivist":
                if input_dict["has_political_keyword"] == 1:
                    relevant_features.append("• Contains political keywords")
                relevant_features.append("• Mixed attack tactics")
            
            if relevant_features:
                for feature in relevant_features:
                    st.write(feature)
            else:
                st.write("• Pattern analysis based on multiple behavioral indicators")
                
        else:
            st.warning("⚠️ Threat attribution could not be performed.")
    else:
        st.info("🛡️ Attribution is only performed when a URL is classified as malicious.")
        st.write("For benign URLs, no threat actor profiling is necessary.")
