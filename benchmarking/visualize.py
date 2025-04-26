import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("benchmark_summary.csv")

df["config_id"] = df["target_model"] + " | " + df["draft_models"]

st.set_page_config(layout="wide")
st.title("🔍 Speculative Decoding Benchmark Dashboard v3")

st.sidebar.header("Filter Options")

prompt_filter = st.sidebar.multiselect("Select Prompt(s):", df["prompt"].unique(), default=list(df["prompt"].unique()))
strategy_filter = st.sidebar.multiselect("Select Strategy(ies):", df["strategy"].unique(), default=list(df["strategy"].unique()))
beam_widths = sorted(df["beam_width"].dropna().unique())
beam_width_filter = st.sidebar.multiselect("Select Beam Width(s):", beam_widths, default=list(beam_widths))
spec_lens = sorted(df["speculative_length"].dropna().unique())
spec_len_filter = st.sidebar.multiselect("Select Speculative Length(s):", spec_lens, default=list(spec_lens))

config_filter = st.sidebar.multiselect("Select Target-Draft Config(s):", df["config_id"].unique(), default=list(df["config_id"].unique()))

filtered_df = df[
    (df["prompt"].isin(prompt_filter)) &
    (df["strategy"].isin(strategy_filter)) &
    (df["speculative_length"].isin(spec_len_filter)) &
    (df["config_id"].isin(config_filter))
]
if "Tree" in strategy_filter:
    filtered_df = filtered_df[(filtered_df["beam_width"].isin(beam_width_filter)) | (filtered_df["beam_width"].isna())]

filtered_df["target_save_ratio"] = filtered_df["tokens_saved"] / filtered_df["tokens_generated"]
filtered_df["acceptance_ratio"] = filtered_df["tokens_accepted"] / filtered_df["tokens_generated"]

st.subheader("Filtered Results")
st.dataframe(filtered_df, use_container_width=True)


# Stacked Bar Chart: Tokens Saved by Strategy
st.subheader("Stacked Bar Chart: Tokens Saved Comparison")
grouping_option = st.selectbox("Group bars by:", ["prompt", "speculative_length"])
bar_data = filtered_df.groupby([grouping_option, "strategy", "config_id"]).agg({"tokens_saved": "sum"}).reset_index()
fig2 = px.bar(
    bar_data,
    x=grouping_option,
    y="tokens_saved",
    color="strategy",
    barmode="group",
    facet_col="config_id",
    title=f"Tokens Saved by {grouping_option.capitalize()} and Config"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Aggregated Statistics by Strategy and Config")
aggs = filtered_df.groupby(["strategy", "config_id"]).agg({
    "tokens_generated": ["mean", "std", "max", "min"],
    "percent_tokens_saved": ["mean", "std", "max", "min"],
    "target_save_ratio": ["mean", "std", "max", "min"],
    "acceptance_ratio": ["mean", "std", "max", "min"],
    "corrections": ["mean", "std", "max", "min"]
}).round(3)
st.dataframe(aggs, use_container_width=True)

st.subheader("Top Configurations (by % Tokens Saved)")
best_configs = filtered_df.sort_values(by="percent_tokens_saved", ascending=False).head(10)
st.table(best_configs)

st.subheader("Target Save Ratio Distribution")
fig3 = px.box(
    filtered_df,
    x="strategy",
    y="target_save_ratio",
    color="config_id",
    points="all",
    title="Target Save Ratio by Strategy and Config"
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("📥 Download Data")
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode(),
    file_name="filtered_benchmark_results.csv",
    mime="text/csv",
)
st.download_button(
    label="Download Aggregated Stats as CSV",
    data=aggs.to_csv().encode(),
    file_name="aggregated_benchmark_stats.csv",
    mime="text/csv",
)

st.success("Full Interactive Comparative Analysis Complete!")
