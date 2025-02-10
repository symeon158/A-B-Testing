import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency

# Streamlit App UI
st.title("A/B Testing Dashboard - AI vs. Rule-Based Recommendations")

# Data Upload
st.sidebar.header("Upload CSV File for A/B Testing")
upload_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

if upload_file:
    df = pd.read_csv(upload_file, encoding='utf-8')
    st.sidebar.success("File uploaded successfully!")

    # Ensure data types are correct
    df["clicks"] = df["clicks"].astype(int)
    df["conversions"] = df["conversions"].astype(int)
    df["retention"] = df["retention"].astype(int)
    df["playtime"] = df["playtime"].astype(float)
    
    # Display Dataset
    st.write("### A/B Testing Dataset Sample")
    st.dataframe(df.head())
    
    # Conduct Hypothesis Testing
    def run_ab_testing(df):
        results = {}

        # 1. T-test for Playtime
        playtime_A = df[df['group'] == 'A']['playtime']
        playtime_B = df[df['group'] == 'B']['playtime']
        t_stat, p_value_ttest = ttest_ind(playtime_A, playtime_B, equal_var=False)
        results["T-Test (Playtime)"] = {"t-statistic": t_stat, "p-value": p_value_ttest}

        # 2. Chi-Square Test for Conversion Rate
        conversion_table = pd.crosstab(df['group'], df['conversions'])
        chi2_stat, p_value_chi2, _, _ = chi2_contingency(conversion_table)
        results["Chi-Square (Conversion Rate)"] = {"chi2-statistic": chi2_stat, "p-value": p_value_chi2}

        # 3. Chi-Square Test for Retention Rate
        retention_table = pd.crosstab(df['group'], df['retention'])
        chi2_ret_stat, p_value_retention, _, _ = chi2_contingency(retention_table)
        results["Chi-Square (Retention Rate)"] = {"chi2-statistic": chi2_ret_stat, "p-value": p_value_retention}

        return results

    # Run A/B Testing Analysis
    st.write("### A/B Testing Results")
    results = run_ab_testing(df)

    for test_name, values in results.items():
        st.write(f"**{test_name}**")
        st.write(f"Statistic: {values['t-statistic' if 'T-Test' in test_name else 'chi2-statistic']:.4f}")
        st.write(f"p-value: {values['p-value']:.6f}")
        if values['p-value'] < 0.05:
            st.success("Statistically Significant (Reject H₀)")
        else:
            st.warning("Not Statistically Significant (Fail to Reject H₀)")
        st.write("---")

    # Visualization
    st.write("### A/B Testing Visualizations")

    # CTR Comparison
    ctr_data = df.groupby("group")["clicks"].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x="group", y="clicks", data=ctr_data, ax=ax)
    ax.set_title("Click-Through Rate (CTR) Comparison")
    ax.set_ylabel("CTR (%)")
    st.pyplot(fig)

    # Conversion Rate Comparison
    cr_data = df.groupby("group")["conversions"].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x="group", y="conversions", data=cr_data, ax=ax)
    ax.set_title("Conversion Rate (CR) Comparison")
    ax.set_ylabel("CR (%)")
    st.pyplot(fig)

    # Retention Rate Comparison
    retention_data = df.groupby("group")["retention"].mean().reset_index()
    fig, ax = plt.subplots()
    sns.barplot(x="group", y="retention", data=retention_data, ax=ax)
    ax.set_title("Retention Rate Comparison")
    ax.set_ylabel("Retention (%)")
    st.pyplot(fig)

    # Playtime Distribution
    st.write("### Playtime Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df[df['group'] == 'A']['playtime'], bins=30, kde=True, label="Group A", color="blue", alpha=0.5)
    sns.histplot(df[df['group'] == 'B']['playtime'], bins=30, kde=True, label="Group B", color="red", alpha=0.5)
    ax.legend()
    ax.set_title("Playtime Distribution (A vs. B)")
    ax.set_xlabel("Playtime (minutes)")
    st.pyplot(fig)

    # Final Decision
    st.write("### 🏆 Final Decision: Deploy AI or Not?")
    if results["Chi-Square (Conversion Rate)"]["p-value"] < 0.05 and results["Chi-Square (Retention Rate)"]["p-value"] < 0.05:
        st.success("🚀 AI-powered recommendations significantly improve user engagement. **Recommendation: Deploy AI Model** ✅")
    else:
        st.warning("⚠️ No significant improvement detected. **Recommendation: Keep Testing or Optimize AI Model** ❌")
else:
    st.sidebar.info("Please upload a CSV file to proceed with A/B testing.")
