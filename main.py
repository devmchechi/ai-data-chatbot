import streamlit as st
from IPython.display import display
import numpy as np
import lux
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import openai
import matplotlib
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import seaborn as sns
import scipy.stats as stats
import datapane as dp
import pandas as pd


def generate_insights_one(df):
    insights = []

    # Summary statistics
    insights.append("Summary Statistics:")
    insights.append(df.describe().to_string())

    # Missing values
    missing_values = df.isnull().sum()
    insights.append("Missing Values:")
    insights.append(missing_values.to_string())

    # Missing values summary
    missing_values_count = df.isnull().sum()
    missing_values_percent = (missing_values_count / len(df)) * 100
    missing_values_summary = pd.DataFrame({
        "Missing Values": missing_values_count,
        "Missing Values %": missing_values_percent
    })
    insights.append("Missing Values Summary:\n")
    insights.append(missing_values_summary.to_string())

    # Data types
    data_types = df.dtypes
    insights.append("Data Types:")
    insights.append(data_types.to_string())

    # Unique values
    unique_values = df.nunique()
    insights.append("Unique Values:")
    insights.append(unique_values.to_string())

    # Correlation matrix
    correlation_matrix = df.corr()
    insights.append("Correlation Matrix:")
    insights.append(correlation_matrix.to_string())

    return "\n\n".join(insights)


def generate_trends_and_patterns_one(df):
    trends_and_patterns = []

    # Distribution of numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col], ax=ax)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        trends_and_patterns.append(fig)

    # Pairwise scatter plots
    sns.set(style="ticks")
    scatter_matrix = sns.pairplot(df, diag_kind="kde")
    trends_and_patterns.append(scatter_matrix.fig)

    return trends_and_patterns


def generate_insights(df):
    insights = []

    # Summary statistics
    summary_stats = df.describe()
    insights.append("Summary Statistics:\n" + summary_stats.to_string())

    # Missing values
    missing_values_count = df.isnull().sum()
    missing_values_percent = (missing_values_count / len(df)) * 100
    missing_values_summary = pd.DataFrame({
        "Missing Values": missing_values_count,
        "Missing Values %": missing_values_percent
    })
    insights.append("Missing Values Summary:\n" +
                    missing_values_summary.to_string())

    # Correlation matrix
    correlation_matrix = df.corr()
    insights.append("Correlation Matrix:\n" + correlation_matrix.to_string())

    return "\n\n".join(insights)


def generate_trends_and_patterns(df):
    trends_and_patterns = []

    # Distribution of numerical columns
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col], ax=ax)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title("Distribution of " + col)
        plt.tight_layout()
        trends_and_patterns.append(fig)

    return trends_and_patterns


def aggregate_data(df, columns):
    aggregated_data = df.groupby(columns).size().reset_index(name='Count')
    return aggregated_data


def generate_profile_report(df):
    profile = ProfileReport(df, explorative=True)
    # To Generate a HTML report file
    profile.to_file("profiling_results.html")
    return profile


def render_sql_view(df):
    view = dp.Blocks(dp.DataTable(df))
    return dp.save_report(view, path="SQL_Rendered_View.html", open=True)


response_history = st.session_state.get("response_history", [])
st.session_state.openai_key = 'OPENAI-API-KEY'
prompt_history = st.session_state.get("prompt_history", [])
st.session_state.df = None
matplotlib.use('TkAgg')

col1, col2, col3 = st.columns(3)

with col1:
    st.image('YOUR-IMAGE', width=250)
with col2:
    st.write(' ')
with col3:
    st.write(' ')

st.title("Conversational Data Chatbot")
    
fig = None

if st.button('New Session'):
    st.session_state.messages = []

if st.session_state.df is None:
    uploaded_file = st.file_uploader(
        "Choose a Single CSV File..",
        type="csv",
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
    
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if prompt := st.chat_input("Enter Question"):
    
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        llm = OpenAI(api_token=st.session_state.openai_key)
        pandas_ai = PandasAI(llm)
        x = pandas_ai(st.session_state.df, prompt=prompt)
            
        if "insights" in prompt.lower():
            insights = generate_insights_one(st.session_state.df)
            st.write(insights)

        elif "trends" in prompt.lower() or "patterns" in prompt.lower():
            trends_and_patterns = generate_trends_and_patterns_one(
                st.session_state.df)
            for fig in trends_and_patterns:
                st.pyplot(fig)

        elif "aggregate" in prompt.lower():
            columns = prompt.lower().split("aggregate ")[1].split(" and ")
            aggregated_data = aggregate_data(st.session_state.df, columns)
            st.subheader("Aggregated Data:")
            st.write(aggregated_data)

        elif "profile" in prompt.lower():
            profile = generate_profile_report(st.session_state.df)
            if profile:
                st.write("Check Profile Report in root directory")
        elif "sql" in prompt.lower() or "SQL" in prompt.lower() or "view" in prompt.lower() or "VIEW" in prompt.lower():
            render_sql_view(st.session_state.df)
            st.write("SQL View Rendered.. Check 'SQL_Rendered_View.html' file")
        elif "histogram" in prompt.lower() or "chart" in prompt.lower() or "bar chart" in prompt.lower() or "line chart" in prompt.lower():
            fig_number = plt.get_fignums()
            if fig_number:
                st.pyplot(plt.gcf())
            else:
                st.write(
                    "Did you get the popup? If not, enable pop-ups and try again!")
        else:
            st.write(x)
        
        st.session_state.messages.append({"role": "assistant", "content": x})
        st.session_state.response_history = response_history
        st.session_state.prompt_history = prompt_history
