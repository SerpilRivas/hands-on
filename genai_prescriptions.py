# genai_prescriptions.py
import google.generativeai as genai
import openai
import requests
import streamlit as st
import json

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    grok_api_key = st.secrets["GROK_API_KEY"]
except (KeyError, FileNotFoundError):
    print("API keys not found in .streamlit/secrets.toml. Some features may be disabled.")
    grok_api_key = None

def get_base_prompt(alert_details):
    return f"""
    You are an expert Security Orchestration, Automation, and Response (SOAR) system.
    A URL has been flagged as a potential phishing attack based on the following characteristics:
    {json.dumps(alert_details, indent=2)}

    Your task is to generate a prescriptive incident response plan.
    Provide your response in a structured JSON format with the following keys:
    - "summary": A brief, one-sentence summary of the threat.
    - "risk_level": A single-word risk level (e.g., "Critical", "High", "Medium").
    - "recommended_actions": A list of specific, technical, step-by-step actions for a security analyst to take.
    - "communication_draft": A brief, professional draft to communicate to the employee who reported the suspicious URL.

    Return ONLY the raw JSON object and nothing else.
    """

def get_gemini_prescription(alert_details):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = get_base_prompt(alert_details)
    # Removing "json" from the start of the string if Gemini includes it
    response_text = model.generate_content(prompt).text.strip().lstrip("```json\n").rstrip("```")
    return json.loads(response_text)

def get_openai_prescription(alert_details):
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    prompt = get_base_prompt(alert_details)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def get_grok_prescription(alert_details):
    if not grok_api_key:
        return {"error": "Grok API key not configured."}
    prompt = get_base_prompt(alert_details)
    url = "https://api.x.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {grok_api_key}", "Content-Type": "application/json"}
    data = {"model": "grok-1", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
    response = requests.post(url, headers=headers, json=data)
    content_str = response.json()['choices'][0]['message']['content']
    return json.loads(content_str.strip().lstrip("```json\n").rstrip("```"))

def generate_prescription(provider, alert_details):
    if provider == "Gemini":
        return get_gemini_prescription(alert_details)
    elif provider == "OpenAI":
        return get_openai_prescription(alert_details)
    elif provider == "Grok":
        return get_grok_prescription(alert_details)
    else:
        raise ValueError("Invalid provider selected")
