import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os


st.set_page_config(layout="wide")
st.sidebar.title("🔍 Speculative Decoding Dashboard")
page = st.sidebar.radio("Go to", ["Benchmark Summary", "Interactive Iteration Viewer"])


if os.path.exists("benchmark_summary.csv"):
    df = pd.read_csv("benchmark_summary.csv")
else:
    st.error("benchmark_summary.csv not found!")
    st.stop()


log_dir = "logs"
log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".json")]
log_data = {}
for f in log_files:
    with open(f, "r") as infile:
        data = json.load(infile)
        key = os.path.basename(f)
        log_data[key] = data


if page == "Benchmark Summary":

    st.title("Speculative Decoding Visualization and Benchmaking Dashboard")

    st.sidebar.header("Filter Options")

    prompt_filter = st.sidebar.multiselect("Select Prompt(s):", df["prompt"].unique(), default=list(df["prompt"].unique()))
    strategy_filter = st.sidebar.multiselect("Select Strategy(ies):", df["strategy"].unique(), default=list(df["strategy"].unique()))
    beam_widths = sorted(df["beam_width"].dropna().unique())
    beam_width_filter = st.sidebar.multiselect("Select Beam Width(s):", beam_widths, default=list(beam_widths))
    spec_lens = sorted(df["speculative_length"].dropna().unique())
    spec_len_filter = st.sidebar.multiselect("Select Speculative Length(s):", spec_lens, default=list(spec_lens))

    df["config_id"] = df["target_model"] + " | " + df["draft_models"]
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

    st.subheader("📊 Tokens Saved Comparison")
    grouping_option = st.selectbox("Group bars by:", ["prompt", "speculative_length"])
    bar_data = filtered_df.groupby([grouping_option, "strategy", "config_id"]).agg({"tokens_saved": "sum"}).reset_index()
    fig2 = px.bar(bar_data, x=grouping_option, y="tokens_saved", color="strategy", barmode="group", facet_col="config_id")
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
    fig3 = px.box(filtered_df, x="strategy", y="target_save_ratio", color="config_id", points="all")
    st.plotly_chart(fig3, use_container_width=True)

    st.download_button("Download Filtered Data", data=filtered_df.to_csv(index=False).encode(), file_name="filtered_benchmark.csv")
    st.download_button("Download Aggregated Stats", data=aggs.to_csv().encode(), file_name="aggregated_stats.csv")

else:
    st.title("📜 Iteration Timeline")

    if not log_data:
        st.warning("No logs found to display timeline.")
    else:
        selected_log = st.selectbox("Select a log file:", [k for k in log_data if k.startswith("specdraft")])
        stats = log_data[selected_log]

        iteration_logs = stats.get("iteration_logs", [])
        draft_models = stats.get("draft_models", [])

        if not iteration_logs:
            st.error("No iteration logs found!")
            st.stop()

        st.markdown("### 📈 Timeline of Drafts, Acceptances and Corrections")

        cumulative_text = []

        for idx, log in enumerate(iteration_logs, 1):
            st.subheader(f"🌀 Iteration {idx}")

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.markdown("#### ✏️ Drafts Proposed")
                for model_name in draft_models:
                    proposed_text = log.get("draft_proposed", {}).get(model_name, "-")
                    st.markdown(f"<span style='color:gray'><b>{model_name}:</b> {proposed_text}</span>", unsafe_allow_html=True)

            with col2:
                st.markdown("#### ✅ Accepted Text")
                for model_name in draft_models:
                    accepted_text = log.get("accepted_text", {}).get(model_name, "-")
                    if accepted_text and accepted_text != "-":
                        st.markdown(f"<span style='color:green'><b>{model_name}:</b> {accepted_text}</span>", unsafe_allow_html=True)

            with col3:
                st.markdown("#### ❌ Correction Applied")
                for model_name in draft_models:
                    corr = log.get("correction", {}).get(model_name, None)
                    if corr:
                        st.markdown(
                            f"<span style='color:red'><b>{model_name}:</b> `{corr['rejected']}` ➔ `{corr['corrected_with']}`</span>",
                            unsafe_allow_html=True
                        )

            st.divider()

            best_model = max(log.get("accepted_text", {}), key=lambda name: len(log["accepted_text"].get(name, "")), default=None)
            if best_model:
                accepted = log["accepted_text"].get(best_model, "")
                if accepted:
                    cumulative_text.append(f"<span style='color:green'>{accepted}</span>")
                corr = log.get("correction", {}).get(best_model, None)
                if corr:
                    cumulative_text.append(f"<span style='color:red'>{corr['corrected_with']}</span>")

        st.subheader("📖 Full Generated Text")
        if cumulative_text:
            full_text_html = "".join(cumulative_text)
            st.markdown(f"<div style='background-color:#f0f2f6;padding:12px;border-radius:8px'>{full_text_html}</div>", unsafe_allow_html=True)
        else:
            st.info("No text generated yet.")

        st.divider()


