import sys, platform
import streamlit as st
import pandas as pd

st.title("✅ Streamlit deploy smoke test")
st.write({"python": sys.version.split()[0],
          "platform": platform.platform(),
          "pandas": pd.__version__})
st.success("If you can see versions above, requirements + Python version are good.")
