<-- Inserted full code here from above for app.py -->
import streamlit as st
import pandas as pd
import openai
import plotly.express as px
from datetime import datetime

# Set your OpenAI API key
openai.api_key = st.secrets.get("openai_api_key") or "YOUR_API_KEY_HERE"

# Streamlit configuration
st.set_page_config(page_title="🪣 S3 Bucket Deletion Advisor", layout="wide", initial_sidebar_state="collapsed")

# Apply dark theme styles
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0e1117;
            color: #e6e6e6;
        }
        .stDataFrame, .stMarkdown, .stTextInput, .stButton {
            color: #e6e6e6;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🪣 S3 Bucket Deletion Advisor")
st.caption("Powered by GPT-4 — Make smart, policy-aware decisions on AWS S3 bucket deletions.")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload your S3 bucket metadata CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    st.subheader("📄 Uploaded Bucket Metadata")
    st.dataframe(df.head(), use_container_width=True)

    results = []
    with st.spinner("🧠 Thinking... Processing rows with GPT-4..."):
        for index, row in df.iterrows():
            try:
                prompt = f"""
You are an intelligent assistant that decides whether an AWS S3 bucket can be deleted.

Rules:
- If total_objects > 0 → Do NOT delete.
- If model version starts with "No Access" and version < 10.0 → Consider deletable.
- If lifecycle contains "Prohibited" → Do NOT delete.
- Buckets older than 6 months with 0 objects and non-prohibited lifecycle → can be deleted.

Now decide: Can the following S3 bucket be deleted?

Bucket details:
- Bucket Name: {row.get('bucket_name', '')}
- Total Objects: {row.get('total_objects', 0)}
- S3 Model Version: {row.get('s3_model_version', '')}
- Lifecycle: {row.get('core_bud_module_lifecycle', '')}
- Environment: {row.get('operating_env', '')}
- Created On: {row.get('creation_date', '')}

Answer format:
Decision: [Yes/No]
Reason: [Explanation]
                """

                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )

                llm_output = response.choices[0].message.content.strip()
                decision_line = llm_output.splitlines()[0]
                reason_line = llm_output.splitlines()[1] if len(llm_output.splitlines()) > 1 else "No reason provided"

                results.append({
                    "Bucket Name": row.get('bucket_name', ''),
                    "Decision": decision_line.replace("Decision:", "").strip(),
                    "Reason": reason_line.replace("Reason:", "").strip()
                })

            except Exception as e:
                results.append({
                    "Bucket Name": row.get('bucket_name', ''),
                    "Decision": "Error",
                    "Reason": str(e)
                })

    result_df = pd.DataFrame(results)

    st.subheader("✅ GPT-4 Decisions")
    st.dataframe(result_df, use_container_width=True)

    # Pie chart summary
    st.subheader("📊 Deletion Summary")
    pie_chart = px.pie(
        result_df[result_df["Decision"].isin(["Yes", "No"])],
        names="Decision",
        title="Can Delete vs Cannot Delete",
        color_discrete_map={"Yes": "green", "No": "red"},
        hole=0.4
    )
    st.plotly_chart(pie_chart, use_container_width=True)

    # Download button
    st.download_button(
        label="⬇️ Download Full Results",
        data=result_df.to_csv(index=False),
        file_name="s3_deletion_decisions_llm.csv",
        mime="text/csv"
    )
else:
    st.info("📥 Please upload a CSV file to begin analysis.")

