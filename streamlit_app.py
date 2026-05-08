"""
Expert Series Trenderizer — Streamlit UI

Read-only dashboard over the committed Parquet data store.
All ON24 / Anthropic calls happen in the ingest pipeline (trenderizer.ingest),
never at request time here.

Tabs:
  1. Top Topics    — bar chart of most-mentioned topics
  2. Rising Topics — topics gaining share (§2.6 algorithm)
  3. Companies     — per-company topic mix over time
  4. Events        — raw event table with question counts
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

# True when running locally with ingest credentials; False on Streamlit Cloud.
IS_LOCAL = bool(os.environ.get("ON24_CLIENT_ID"))
DATA_DIR = BASE_DIR / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"

st.set_page_config(
    page_title="Expert Series Trenderizer",
    page_icon="📈",
    layout="wide",
)

# ── Auth ──────────────────────────────────────────────────────────────────────
try:
    _PW = st.secrets["APP_PASSWORD"]
except Exception:
    _PW = os.environ.get("APP_PASSWORD", "s1emens")

if not st.session_state.get("authed"):
    with st.form("login"):
        st.subheader("Sign in")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    if submitted:
        if pw == _PW:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=120)
def _load_table(name: str) -> pd.DataFrame:
    base = DATA_DIR / name
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception:
            pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@st.cache_data(ttl=120)
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _parse_tags(raw) -> list[str]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return []


# ── Rising Topics algorithm (spec §2.6) ───────────────────────────────────────

def compute_rising_topics(df: pd.DataFrame, min_mentions: int = 5) -> pd.DataFrame:
    """
    Split the window in half, compute topic-share delta (recent – prior).
    Rank by delta descending, show top 20 with min_mentions filter.
    """
    if df.empty or "topic_label" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["topic_label"]).copy()
    if df.empty:
        return pd.DataFrame()

    ts = "created_ts_pacific"
    if ts not in df.columns:
        return pd.DataFrame()

    df[ts] = pd.to_datetime(df[ts], utc=True, errors="coerce")
    df = df.dropna(subset=[ts])
    if df.empty:
        return pd.DataFrame()

    t_min, t_max = df[ts].min(), df[ts].max()
    mid = t_min + (t_max - t_min) / 2

    prior = df[df[ts] < mid]
    recent = df[df[ts] >= mid]
    total_counts = df["topic_label"].value_counts()

    def shares(subset: pd.DataFrame) -> pd.Series:
        c = subset["topic_label"].value_counts()
        return c / c.sum() if len(c) else c

    prior_s = shares(prior)
    recent_s = shares(recent)
    eps = 1e-9

    rows = []
    for topic in set(prior_s.index) | set(recent_s.index):
        total = int(total_counts.get(topic, 0))
        if total < min_mentions:
            continue
        sr = float(recent_s.get(topic, 0.0))
        sp = float(prior_s.get(topic, 0.0))
        delta = sr - sp
        rows.append({
            "Topic": topic,
            "Recent %": round(sr * 100, 1),
            "Prior %": round(sp * 100, 1),
            "Δ (pp)": round(delta * 100, 1),
            "Lift": round(sr / max(sp, eps), 1),
            "Mentions": total,
        })

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("Δ (pp)", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )


# ── Load data ─────────────────────────────────────────────────────────────────

events_df = _load_table("events")
questions_df = _load_table("questions")

if not events_df.empty and "tags" in events_df.columns:
    events_df["tags"] = events_df["tags"].apply(_parse_tags)

if not questions_df.empty and "created_ts_pacific" in questions_df.columns:
    questions_df["created_ts_pacific"] = pd.to_datetime(
        questions_df["created_ts_pacific"], utc=True, errors="coerce"
    )

if not events_df.empty and "start_ts_pacific" in events_df.columns:
    events_df["start_ts_pacific"] = pd.to_datetime(
        events_df["start_ts_pacific"], utc=True, errors="coerce"
    )


# ── Header ────────────────────────────────────────────────────────────────────

st.title("Expert Series Trenderizer")
st.caption("What are customers asking about? ON24 Q&A data enriched with Claude Haiku.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Filters")

    # Date range
    if not questions_df.empty and "created_ts_pacific" in questions_df.columns:
        valid_ts = questions_df["created_ts_pacific"].dropna()
        default_min = valid_ts.min().date() if not valid_ts.empty else datetime.now().date() - timedelta(days=365)
        default_max = valid_ts.max().date() if not valid_ts.empty else datetime.now().date()
    else:
        default_min = datetime.now().date() - timedelta(days=365)
        default_max = datetime.now().date()

    date_range = st.date_input(
        "Date range",
        value=(default_min, default_max),
        min_value=default_min,
        max_value=default_max,
    )

    # Product multiselect
    product_options: list[str] = []
    if not events_df.empty and "inferred_product" in events_df.columns:
        product_options = sorted(events_df["inferred_product"].dropna().unique().tolist())
    selected_products = st.multiselect("Products", options=product_options)

    # Company multiselect (only when company data is available)
    company_options: list[str] = []
    has_company_data = (
        not questions_df.empty and "inferred_company" in questions_df.columns
        and questions_df["inferred_company"].notna().any()
    )
    if has_company_data:
        company_options = sorted(
            questions_df.loc[
                questions_df["inferred_company"].notna()
                & (questions_df["inferred_company"] != "Personal / Unknown"),
                "inferred_company",
            ].unique().tolist()
        )
    selected_companies = st.multiselect("Companies", options=company_options,
                                        disabled=not has_company_data)

    include_personal = st.toggle("Include personal-email askers", value=False)
    include_siemens = st.toggle("Include Siemens employees", value=False)

    st.divider()
    if IS_LOCAL:
        if st.button("Refresh data", type="primary", use_container_width=True):
            with st.spinner("Running ingest pipeline..."):
                proc = subprocess.run(
                    [sys.executable, "-m", "trenderizer.ingest"],
                    capture_output=True,
                    text=True,
                    cwd=str(BASE_DIR),
                )
            if proc.returncode == 0:
                st.success("Ingest complete.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Ingest failed:\n```\n{proc.stderr[-600:]}\n```")
    else:
        st.caption("Data is updated by the maintainer and pushed to the repository.")


# ── Apply filters ─────────────────────────────────────────────────────────────

def apply_filters(q: pd.DataFrame, e: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if q.empty:
        return q, e

    q = q.copy()

    if len(date_range) == 2 and "created_ts_pacific" in q.columns:
        start, end = date_range
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        q = q[
            (q["created_ts_pacific"] >= start_ts)
            & (q["created_ts_pacific"] < end_ts)
        ]

    if not include_siemens and "is_siemens_employee" in q.columns:
        q = q[~q["is_siemens_employee"].fillna(False)]

    if not include_personal and "inferred_company" in q.columns:
        q = q[q["inferred_company"] != "Personal / Unknown"]

    if selected_products and not e.empty and "inferred_product" in e.columns:
        valid_eids = e.loc[e["inferred_product"].isin(selected_products), "event_id"]
        q = q[q["event_id"].isin(valid_eids)]

    if selected_companies and "inferred_company" in q.columns:
        q = q[q["inferred_company"].isin(selected_companies)]

    return q, e


q_f, e_f = apply_filters(questions_df, events_df)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Top Topics", "Rising Topics", "Companies", "Events Explorer"
])


# ════════════════════════════════════════════════════════════════════════════════
# Tab 1 — Top Topics
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Top Topics")

    if q_f.empty or "topic_label" not in q_f.columns or q_f["topic_label"].dropna().empty:
        st.info("No topic data for the selected filters. Run an ingest to populate the store.")
    else:
        topic_counts = (
            q_f.dropna(subset=["topic_label"])
            .groupby("topic_label")
            .size()
            .reset_index(name="Mentions")
            .sort_values("Mentions", ascending=False)
            .head(25)
        )

        if topic_counts.empty:
            st.info("No topics found.")
        else:
            try:
                import plotly.express as px

                # Attach dominant product for color coding
                if not events_df.empty and "inferred_product" in events_df.columns:
                    q_prod = q_f.merge(
                        events_df[["event_id", "inferred_product"]].drop_duplicates(),
                        on="event_id",
                        how="left",
                    )
                    dominant = (
                        q_prod.dropna(subset=["topic_label", "inferred_product"])
                        .groupby(["topic_label", "inferred_product"])
                        .size()
                        .reset_index(name="n")
                        .sort_values("n", ascending=False)
                        .groupby("topic_label")
                        .first()
                        .reset_index()
                        .rename(columns={"inferred_product": "Product"})
                    )
                    topic_counts = topic_counts.merge(
                        dominant[["topic_label", "Product"]], on="topic_label", how="left"
                    )
                else:
                    topic_counts["Product"] = "Unknown"

                fig = px.bar(
                    topic_counts,
                    x="Mentions",
                    y="topic_label",
                    color="Product",
                    orientation="h",
                    labels={"topic_label": "Topic"},
                    height=max(400, len(topic_counts) * 26),
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

            except ImportError:
                st.dataframe(topic_counts, use_container_width=True, hide_index=True)

            # Drill-down
            st.divider()
            drill_topic = st.selectbox(
                "Drill into a topic",
                options=["(select a topic)"] + topic_counts["topic_label"].tolist(),
                key="tab1_drill",
            )
            if drill_topic != "(select a topic)":
                drill_df = q_f[q_f["topic_label"] == drill_topic].copy()
                if not events_df.empty:
                    drill_df = drill_df.merge(
                        events_df[["event_id", "title", "inferred_product"]].drop_duplicates(),
                        on="event_id",
                        how="left",
                    )
                display_cols = {
                    "created_ts_pacific": "Date",
                    "inferred_company": "Company",
                    "title": "Event",
                    "content": "Question",
                }
                available = {k: v for k, v in display_cols.items() if k in drill_df.columns}
                st.dataframe(
                    drill_df[list(available.keys())].rename(columns=available),
                    use_container_width=True,
                    hide_index=True,
                )


# ════════════════════════════════════════════════════════════════════════════════
# Tab 2 — Rising Topics
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Rising Topics")
    st.caption(
        "Topics whose share of mentions grew in the **second half** of the selected window "
        "vs the first half. Ranked by percentage-point gain. "
        "Algorithm: split window in half, compute share delta (recent − prior), filter to "
        "topics with at least *N* total mentions."
    )

    if q_f.empty:
        st.info("No data for selected filters.")
    else:
        min_mentions = st.slider("Minimum total mentions", min_value=1, max_value=30, value=5, key="rising_min")
        rising = compute_rising_topics(q_f, min_mentions=min_mentions)

        if rising.empty:
            st.info(
                "Not enough data to compute rising topics. "
                "Try expanding the date range or lowering the minimum-mentions threshold."
            )
        else:
            rising_selection = st.dataframe(
                rising,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "Recent %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Prior %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Δ (pp)": st.column_config.NumberColumn(format="%+.1f pp"),
                },
            )

            selected_rows = (rising_selection.selection.rows
                             if hasattr(rising_selection, "selection") else [])
            if selected_rows:
                sel_topic = rising.iloc[selected_rows[0]]["Topic"]
                topic_qs = q_f[q_f["topic_label"] == sel_topic].copy()

                st.divider()
                st.subheader(f"Questions — {sel_topic}")
                if topic_qs.empty:
                    st.info("No questions for this topic under the current filters.")
                else:
                    if not events_df.empty:
                        topic_qs = topic_qs.merge(
                            events_df[["event_id", "title"]].drop_duplicates(),
                            on="event_id", how="left",
                        )
                    q_cols = {
                        "topic_label": "Topic",
                        "content": "Question",
                        "inferred_company": "Company",
                        "title": "Event",
                    }
                    available_q = {k: v for k, v in q_cols.items() if k in topic_qs.columns}
                    st.dataframe(
                        topic_qs[list(available_q.keys())].rename(columns=available_q),
                        use_container_width=True,
                        hide_index=True,
                    )


# ════════════════════════════════════════════════════════════════════════════════
# Tab 3 — Companies
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Per-Company Topic Trends")

    if q_f.empty or "inferred_company" not in q_f.columns or not q_f["inferred_company"].notna().any():
        st.info(
            "Company data is not available in this deployment. "
            "Run the ingest pipeline locally to enrich company names, "
            "then update the data files in the repository."
        )
    else:
        company_list = sorted(
            q_f.loc[
                q_f["inferred_company"].notna()
                & (q_f["inferred_company"] != "Personal / Unknown"),
                "inferred_company",
            ].unique().tolist()
        )

        if not company_list:
            st.info("No named companies found in the selected window.")
        else:
            chosen = st.selectbox("Select a company", options=company_list, key="company_select")
            co_df = q_f[q_f["inferred_company"] == chosen].copy()

            if co_df.empty:
                st.info(f"No questions from {chosen} in this window.")
            else:
                co_df["month"] = co_df["created_ts_pacific"].dt.to_period("M").astype(str)

                pivot = (
                    co_df.dropna(subset=["topic_label"])
                    .groupby(["month", "topic_label"])
                    .size()
                    .reset_index(name="count")
                )

                if not pivot.empty:
                    try:
                        import plotly.express as px
                        fig = px.area(
                            pivot,
                            x="month",
                            y="count",
                            color="topic_label",
                            title=f"{chosen} — Topic Mix Over Time",
                            labels={"month": "Month", "count": "Questions", "topic_label": "Topic"},
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        st.dataframe(pivot, use_container_width=True, hide_index=True)

                st.subheader(f"{chosen} — All Questions ({len(co_df)})")
                display = co_df.copy()
                if not events_df.empty:
                    display = display.merge(
                        events_df[["event_id", "title"]].drop_duplicates(),
                        on="event_id",
                        how="left",
                    )
                cols = {
                    "created_ts_pacific": "Date",
                    "topic_label": "Topic",
                    "title": "Event",
                    "content": "Question",
                }
                available = {k: v for k, v in cols.items() if k in display.columns}
                st.dataframe(
                    display[list(available.keys())].rename(columns=available),
                    use_container_width=True,
                    hide_index=True,
                )


# ════════════════════════════════════════════════════════════════════════════════
# Tab 4 — Events Explorer
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Events Explorer")

    if events_df.empty:
        st.info("No events found. Run `python -m trenderizer.ingest --backfill` to populate.")
    else:
        e_display = events_df.copy()
        if selected_products and "inferred_product" in e_display.columns:
            e_display = e_display[e_display["inferred_product"].isin(selected_products)]

        # Question counts per event
        if not questions_df.empty and "event_id" in questions_df.columns:
            qcounts = questions_df.groupby("event_id").size().reset_index(name="Questions")
            e_display = e_display.merge(qcounts, on="event_id", how="left")
            e_display["Questions"] = e_display["Questions"].fillna(0).astype(int)

            # Customer questions (non-Siemens)
            if "is_siemens_employee" in questions_df.columns:
                cust_counts = (
                    questions_df[~questions_df["is_siemens_employee"].fillna(False)]
                    .groupby("event_id")
                    .size()
                    .reset_index(name="Customer Q")
                )
                e_display = e_display.merge(cust_counts, on="event_id", how="left")
                e_display["Customer Q"] = e_display["Customer Q"].fillna(0).astype(int)

        display_cols = {
            "event_id": "Event ID",
            "start_ts_pacific": "Date",
            "series": "Series",
            "inferred_product": "Product",
            "Customer Q": "Customer Q",
            "Questions": "Total Q",
            "title": "Title",
        }
        available = {k: v for k, v in display_cols.items() if k in e_display.columns}

        e_sorted = (
            e_display.sort_values("start_ts_pacific", ascending=False)
            .reset_index(drop=True)
        )
        e_sorted_ids = e_sorted["event_id"].tolist()

        event_selection = st.dataframe(
            e_sorted[list(available.keys())].rename(columns=available),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        selected_rows = (event_selection.selection.rows
                         if hasattr(event_selection, "selection") else [])
        if selected_rows:
            sel_event_id = e_sorted_ids[selected_rows[0]]
            sel_title = events_df.loc[
                events_df["event_id"] == sel_event_id, "title"
            ].iloc[0] if not events_df.empty else f"Event {sel_event_id}"

            event_qs = q_f[q_f["event_id"] == sel_event_id].copy()

            st.divider()
            st.subheader(f"Questions — {sel_title}")
            if event_qs.empty:
                st.info("No questions for this event under the current filters.")
            else:
                q_cols = {
                    "topic_label": "Topic",
                    "content": "Question",
                    "inferred_company": "Company",
                }
                available_q = {k: v for k, v in q_cols.items() if k in event_qs.columns}
                st.dataframe(
                    event_qs[list(available_q.keys())].rename(columns=available_q),
                    use_container_width=True,
                    hide_index=True,
                )


# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
manifest = load_manifest()
last_run = manifest.get("last_run_at")
total_events = len(events_df) if not events_df.empty else 0
total_questions = len(questions_df) if not questions_df.empty else 0

if last_run:
    try:
        last_dt = datetime.fromisoformat(last_run.replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - last_dt
        if age.total_seconds() < 3600:
            freshness = f"{int(age.total_seconds() / 60)} min ago"
        elif age.total_seconds() < 86400:
            freshness = f"{int(age.total_seconds() / 3600)} hr ago"
        else:
            freshness = f"{age.days} day(s) ago"
        color = "green" if age.days < 2 else "orange" if age.days < 7 else "red"
        st.caption(
            f":{color}[Last ingest: {freshness}] · "
            f"{total_events} events · {total_questions} questions total"
        )
    except Exception:
        st.caption(f"Last ingest: {last_run} · {total_events} events · {total_questions} questions")
else:
    st.caption(
        ":red[No ingest run yet.] "
        "Run `python -m trenderizer.ingest --backfill` to populate the data store."
    )
