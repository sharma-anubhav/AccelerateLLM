#!/usr/bin/env python
import streamlit as st
import pandas as pd
import plotly.express as px
import json, os
from pathlib import Path
from streamlit_option_menu import option_menu

st.set_page_config(page_title="AccelerateLLM: Faster Inference & Profiling", layout="wide")

primary = st.get_option("theme.primaryColor")

text_color = st.get_option("theme.textColor")

st.markdown(
    f"""
    <div style="text-align:center; padding: 1rem 0;">
      <h1 style="margin:0; color:{text_color}; font-size:2.5rem;">
        AccelerateLLM: Faster Inference & Profiling
      </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    div[data-testid="option-menu"] .container {
        border-radius: 12px;
        background: linear-gradient(135deg,#1f2937 0%,#111827 100%);
        box-shadow: 0 4px 12px rgba(0,0,0,.28);
        margin-bottom: 1.2rem;
    }
    div[data-testid="option-menu"] .nav {margin:0;gap:.25rem;}
    /* unselected links */
    div[data-testid="option-menu"] .nav-link {
        color:#d1d5db !important;             
        font-weight:500;
        border-radius: 10px;
        transition: all .2s;
        padding:10px 20px;
    }
    div[data-testid="option-menu"] .nav-link:hover {
        background: rgba(46,204,113,.10);
        color:#ecfdf5 !important;
    }
    div[data-testid="option-menu"] .nav-link.active {
        background: #2ecc71 !important;
        color:#ffffff !important;
        box-shadow: 0 2px 8px rgba(46,204,113,.45);
    }
    div[data-testid="option-menu"] .nav-link i {
        font-size: 19px !important;
        margin-right:6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

selected = option_menu(
    menu_title=None,
    options=["Home", "Profiling", "Benchmark"],
    icons=["house-fill", "bar-chart-line-fill", "clock-history"],
    default_index=0,
    orientation="horizontal",
    styles={          
        "container": {"padding": "0"},
        "nav-link-selected": {}               
    },
)

tab_home     = selected == "Home"
tab_bench    = selected == "Benchmark"
tab_iter = selected == "Profiling"

if tab_home:
    README_PATH = Path(__file__).parent.parent / "README.md"  
    if not README_PATH.exists():
        st.error(f"README.md not found at {README_PATH}")
    else:
        st.markdown(README_PATH.read_text(encoding="utf-8"), unsafe_allow_html=True)

# ╭───────────────────────────────────────────╮
# │  📊  BENCHMARK SUMMARY                    │
# ╰───────────────────────────────────────────╯
elif tab_bench:
    # ── Load CSV ────────────────────────────────────────────────
    DATA_PATH = Path(__file__).parent / "benchmark_summary.csv"
    if not DATA_PATH.exists():
        st.error("benchmark_summary.csv not found in the app directory!")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    with st.expander("🔧  Filter Options", expanded=True):

        prompt_filter = st.multiselect(
            "Prompt (type to search)",
            options=df["prompt"].unique(),
            default=list(df["prompt"].unique()),
            key="prompt_filter",
        )

        col_left, col_right = st.columns([2,3], gap="medium")
        with col_left:
            st.markdown("**Strategy**")
            strat_opts = df["strategy"].unique().tolist()
            strat_cols = st.columns(len(strat_opts))
            strategy_filter = []
            for i, strat in enumerate(strat_opts):
                checked = strat_cols[i].checkbox(
                    strat,
                    value=True,
                    key=f"strat_chk_{strat}"
                )
                if checked:
                    strategy_filter.append(strat)

        with col_right:
            df["config_id"] = df["target_model"] + " | " + df["draft_models"]
            cfg_filter = st.multiselect(
                "Target ▸ Draft Config",
                options=df["config_id"].unique(),
                default=list(df["config_id"].unique()),
                key="cfg_filter",
            )
        col_beam, col_spec = st.columns([3, 2], gap="medium")
        with col_beam:
            beam_widths = sorted(df["beam_width"].dropna().unique())
            beam_filter = st.multiselect(
                "Beam Width (Tree-only)",
                options=beam_widths,
                default=beam_widths,
                key="beam_filter",
            )
        with col_spec:
            spec_lens = sorted(df["speculative_length"].dropna().unique())
            spec_filter = st.multiselect(
                "Spec Length",
                options=spec_lens,
                default=spec_lens,
                key="spec_len_filter",
            )

        if st.button("↺  Reset all filters"):
            st.session_state.prompt_filter   = list(df["prompt"].unique())
            st.session_state.strategy_filter = list(df["strategy"].unique())
            st.session_state.cfg_filter      = list(df["config_id"].unique())
            st.session_state.beam_filter     = beam_widths
            st.session_state.spec_len_filter = spec_lens
            st.experimental_rerun()

    filt = (
        df["prompt"].isin(prompt_filter) &
        df["strategy"].isin(strategy_filter) &
        df["speculative_length"].isin(spec_filter) &
        df["config_id"].isin(cfg_filter)
    )
    if "Tree" in strategy_filter:
        filt &= (df["beam_width"].isin(beam_filter) | df["beam_width"].isna())

    filtered_df = df[filt].copy()
    filtered_df["target_save_ratio"] = filtered_df["tokens_saved"] / filtered_df["tokens_generated"]
    filtered_df["acceptance_ratio"]  = filtered_df["tokens_accepted"] / filtered_df["tokens_generated"]

    st.subheader("Filtered Results")
    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("📊 Tokens Saved Comparison")
    group_by = st.selectbox("Group bars by:", ["prompt", "speculative_length"])
    bar_data = (
        filtered_df
        .groupby([group_by, "strategy", "config_id"])
        .agg(tokens_saved=("tokens_saved", "sum"))
        .reset_index()
    )
    st.plotly_chart(
        px.bar(bar_data, x=group_by, y="tokens_saved",
               color="strategy", barmode="group", facet_col="config_id"),
        use_container_width=True,
    )

    st.subheader("Aggregated Statistics by Strategy and Config")
    aggs = (
        filtered_df
        .groupby(["strategy", "config_id"])
        .agg({
            "tokens_generated": ["mean", "std", "max", "min"],
            "percent_tokens_saved": ["mean", "std", "max", "min"],
            "target_save_ratio": ["mean", "std", "max", "min"],
            "acceptance_ratio": ["mean", "std", "max", "min"],
            "corrections": ["mean", "std", "max", "min"]
        })
        .round(3)
    )
    st.dataframe(aggs, use_container_width=True)

    st.subheader("Top Configurations (by % Tokens Saved)")
    best_configs = filtered_df.sort_values("percent_tokens_saved", ascending=False).head(10)
    st.table(best_configs)

    st.subheader("Target Save Ratio Distribution")
    st.plotly_chart(
        px.box(filtered_df, x="strategy", y="target_save_ratio",
               color="config_id", points="all"),
        use_container_width=True,
    )

    st.download_button("Download Filtered Data",
                       data=filtered_df.to_csv(index=False).encode(),
                       file_name="filtered_benchmark.csv")
    st.download_button("Download Aggregated Stats",
                       data=aggs.to_csv().encode(),
                       file_name="aggregated_stats.csv")


elif tab_iter:
    st.title("📈 Timeline of Drafts, Acceptances and Corrections")

    LOG_PATH = Path(__file__).parent / "logs"
    if not LOG_PATH.exists():
        st.warning("logs/ directory not found"); st.stop()

    log_files = [f for f in LOG_PATH.iterdir() if f.suffix == ".json"]
    if not log_files:
        st.warning("No JSON logs detected"); st.stop()

    log_choice = st.selectbox(
        "Choose a log:",
        [f.name for f in log_files if f.name.startswith("specdraft")]
    )
    stats = json.loads(Path(LOG_PATH, log_choice).read_text())
    iteration_logs = stats.get("iteration_logs", [])
    draft_models   = stats.get("draft_models", [])
    if not iteration_logs:
        st.error("No iteration logs found"); st.stop()

    st.markdown("#### 🔧 Configuration")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target Model", stats["target_model"].split("/")[-1])
    c2.metric("Draft Length", stats["draft_length"])
    c3.metric("Max New Tokens", stats["max_new_tokens"])
    c4.metric("Iterations", stats["metrics"]["total_iterations"])

    m = stats["metrics"]
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Tokens Generated", m["tokens_generated"])
    o2.metric("Tokens Accepted", m["draft_tokens_accepted"])
    o3.metric("Corrections", m["corrections_by_target"])
    o4.metric("% Saved", f"{m['percent_tokens_saved']}%")

    st.divider()
    st.markdown(
        "**Legend:**  "
        "<span style='color:green'>● Green</span> = draft-accepted text &nbsp;&nbsp; "
        "<span style='color:red'>● Red</span> = corrections applied by target",
        unsafe_allow_html=True
    )

    cumulative_text = []
    for log in iteration_logs:
        best = max(
            log.get("accepted_text", {}),
            key=lambda name: len(log["accepted_text"].get(name, "")),
            default=None
        )
        if best:
            acc = log["accepted_text"].get(best, "")
            if acc:
                cumulative_text.append(f"<span style='color:green'>{acc}</span>")
            corr = log.get("correction", {}).get(best)
            if corr:
                cumulative_text.append(f"<span style='color:red'>{corr['corrected_with']}</span>")

    st.subheader("📖 Full Generated Text")
    if cumulative_text:
        st.markdown(
            f"<div style='background:#f0f2f6;padding:12px;border-radius:8px'>"
            f"{''.join(cumulative_text)}</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("No text generated yet.")

    st.divider()

    for idx, log in enumerate(iteration_logs, 1):
        st.subheader(f"Iteration {idx}")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("#### ✏️ Drafts Proposed")
            for m in draft_models:
                txt = log.get("draft_proposed", {}).get(m, "-")
                st.markdown(f"<span style='color:gray'><b>{m}:</b> {txt}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### ✅ Accepted Text")
            for m in draft_models:
                acc = log.get("accepted_text", {}).get(m, "-")
                if acc and acc != "-":
                    st.markdown(f"<span style='color:green'><b>{m}:</b> {acc}</span>", unsafe_allow_html=True)

        with col3:
            st.markdown("#### ❌ Corrections Applied")
            for m in draft_models:
                corr = log.get("correction", {}).get(m)
                if corr:
                    st.markdown(
                        f"<span style='color:red'><b>{m}:</b> `{corr['rejected']}` ➔ `{corr['corrected_with']}`</span>",
                        unsafe_allow_html=True
                    )

        st.divider()
    m = stats["metrics"]
    st.markdown("#### 🧮 Model Metrics (Drafts vs. Target)")
    model_stats = [
        {"name": stats["target_model"].split("/")[-1],
         "key1": "Calls",  "val1": m["actual_target_calls"],
         "key2": "Saves",  "val2": m["tokens_saved"]}
    ] + [
        {"name": dm["name"].split("/")[-1],
         "key1": "Proposed", "val1": dm["proposed"],
         "key2": "Accepted", "val2": dm["accepted"]}
        for dm in m["per_draft_metrics"]
    ]
    cols = st.columns(len(model_stats))
    for col, ms in zip(cols, model_stats):
        col.markdown(f"**{ms['name']}**")
        col.metric(ms["key1"], ms["val1"])
        col.metric(ms["key2"], ms["val2"])
