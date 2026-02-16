import re
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Developer Journey Intelligence", layout="wide")

st.markdown(
    "<meta name='color-scheme' content='dark'>",
    unsafe_allow_html=True
)

# Use the exact CSV filename as it exists in your GitHub repo:
DATA_FILE = "Syntasso Website Data (1).csv"

DISPLAY_COLS = [
    "activity_date", "activity_type", "fname", "lname",
    "linkedin_url", "org_hq_country", "city", "state", "company_name",
    "company_domain", "industry", "normalised_employee_range", "normalised_annual_revenue"
]


# =========================
# STYLING (FIXED FOR DARK THEME)
# =========================
st.markdown(
    """
<style>
    /* ---- Page header ---- */
    .main-header {
        font-size: 28px;
        font-weight: 800;
        color: #E5E7EB;              /* visible on dark background */
        margin-bottom: 18px;
        border-bottom: 2px solid #374151;
        padding-bottom: 10px;
    }

    /* ---- Metric Cards ---- */
    .metric-row { display: flex; gap: 15px; margin-bottom: 18px; }
    .metric-card {
        flex: 1; background: #ffffff; padding: 18px; border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25), 0 2px 4px -1px rgba(0, 0, 0, 0.20);
        border-left: 5px solid #2563EB; text-align: center;
    }
    .metric-val { font-size: 32px; font-weight: 800; color: #111827; }
    .metric-lbl { font-size: 13px; font-weight: 700; color: #4B5563; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 5px; }

    /* ---- Filter Panel ---- */
    .filter-panel {
        background: rgba(255,255,255,0.06);
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .filter-title {
        font-size: 14px;
        font-weight: 800;
        color: #E5E7EB;              /* visible on dark background */
        margin-bottom: 10px;
    }

    /* ---- Tables (DARK, readable) ---- */
    .table-container {
        overflow-x: auto;
        overflow-y: hidden;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.12);
        margin-top: 8px;
        background: #0B1220;
    }
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        min-width: 1100px;
    }
    .styled-table thead tr {
        background-color: #111827;
        color: #FFFFFF;
        text-align: left;
    }
    .styled-table th, .styled-table td {
        padding: 10px 12px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        white-space: nowrap;
    }
    .styled-table tbody tr:nth-of-type(odd)  { background-color: #0B1220; }
    .styled-table tbody tr:nth-of-type(even) { background-color: #0F172A; }
    .styled-table tbody tr:hover { background-color: #111827; }
    .styled-table tbody td { color: #E5E7EB; }       /* readable text */
    .styled-table a { color: #60A5FA; text-decoration: none; }
    .styled-table a:hover { text-decoration: underline; }

    /* Make Streamlit widget labels readable in dark theme */
    .stMarkdown, .stTextLabel, label, p, span { color: #E5E7EB; }

</style>
""",
    unsafe_allow_html=True
)

st.markdown("<div class='main-header'>🚀 Developer Journey Intelligence</div>", unsafe_allow_html=True)


# =========================
# HELPERS
# =========================
def make_sessions(df: pd.DataFrame, user_col: str, time_col: str, threshold_minutes: int = 30) -> pd.DataFrame:
    df = df.sort_values([user_col, time_col]).copy()
    df["time_diff"] = df.groupby(user_col)[time_col].diff()
    df["new_sess"] = (df["time_diff"] > timedelta(minutes=threshold_minutes)) | (df["time_diff"].isnull())
    df["sess_id"] = df.groupby(user_col)["new_sess"].cumsum().astype(str)
    df["uniq_sess"] = df[user_col].astype(str) + "_" + df["sess_id"]
    return df


def get_category(row) -> str:
    u = str(row.get("url", "")).lower()
    h1 = str(row.get("h1", "")).lower()
    if "kratix" in u or "kratix" in h1:
        if "doc" in u or "start" in u:
            return "Kratix Docs"
        return "Kratix Website"
    if "syntasso" in u or "syntasso" in h1:
        if "doc" in u or "start" in u:
            return "Syntasso Docs"
        return "Syntasso Website"
    return "Other Source"


def get_detail(row) -> str:
    u = str(row.get("url", "")).lower()
    if "pricing" in u:
        return "Pricing"
    if "blog" in u:
        return "Blog"
    if "workshop" in u:
        return "Workshop"
    if "install" in u:
        return "Installation"
    if "doc" in u:
        return "Docs (Read)"
    if "about" in u or "team" in u:
        return "About/Company"
    if "feature" in u or "product" in u:
        return "Product Features"
    if "case" in u or "customer" in u:
        return "Use Cases"
    if u.endswith(".io/") or u == "https://www.syntasso.io/":
        return "Homepage"
    return "General Pages"


def get_page_name(row) -> str:
    h1 = str(row.get("h1", "")).strip()
    if h1 and h1.lower() not in ("nan", ""):
        return h1

    u = str(row.get("url", ""))
    try:
        u = re.sub(r"https?://(www\.)?", "", u)
        u = u.split("?")[0]
        parts = u.split("/")
        if len(parts) > 1 and parts[1].strip() != "":
            path = " ".join(parts[1:])
            return path.replace("-", " ").replace("_", " ").title()
        return "Homepage"
    except Exception:
        return "Unknown Page"


@st.cache_data(show_spinner=False)
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    if "activity_date" in df.columns:
        df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")

    # Identity (Anonymous devs) -> dev_handle
    if "dev_handle" in df.columns:
        df["uid"] = df["dev_handle"]
    else:
        df["uid"] = df.index

    # Deanonymized identity (unique reo_id/reo_dev_id)
    if "reo_dev_id" in df.columns:
        df["deanon_id"] = df["reo_dev_id"]
    elif "reo_id" in df.columns:
        df["deanon_id"] = df["reo_id"]
    else:
        df["deanon_id"] = pd.NA

    # Deanonymized Check (Strict)
    df["is_deanonymized"] = False
    df.loc[df["deanon_id"].notna(), "is_deanonymized"] = True
    if "linkedin_url" in df.columns:
        df.loc[df["linkedin_url"].notna(), "is_deanonymized"] = True

    # Sessions
    if "activity_date" in df.columns:
        df = make_sessions(df, "uid", "activity_date", threshold_minutes=30)
    else:
        df["uniq_sess"] = df["uid"].astype(str) + "_1"

    # Categories & page labels
    df["cat_entry"] = df.apply(get_category, axis=1)
    df["cat_detail"] = df.apply(get_detail, axis=1)
    df["page_name"] = df.apply(get_page_name, axis=1)

    return df


def plot_sankey_detailed(df: pd.DataFrame) -> go.Figure:
    df = df.sort_values(["uniq_sess", "activity_date"])
    high_intent = ["cli", "copy_command", "form_filled", "sign_up"]

    grp = df.groupby("uniq_sess")
    start = grp.first()["cat_entry"].reset_index(name="step1")
    mid = grp["cat_detail"].agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]).reset_index(name="step2")

    def get_outcome(x: pd.Series) -> str:
        acts = set(x.astype(str).str.lower())
        if any(h in acts for h in high_intent):
            return "High Intent Action"
        if len(x) == 1:
            return "Bounced"
        return "Browsed & Left"

    end = grp["activity_type"].apply(get_outcome).reset_index(name="step3")

    journey = start.merge(mid, on="uniq_sess").merge(end, on="uniq_sess")

    bounce_rates = journey[journey["step3"] == "Bounced"].groupby("step1").size()
    total_rates = journey.groupby("step1").size()

    label_map = {}
    for n in total_rates.index:
        b = int(bounce_rates.get(n, 0))
        t = int(total_rates.get(n, 1))
        label_map[n] = f"{n} ({int(b / t * 100)}% Drop-off)"

    journey["step1_lbl"] = journey["step1"].map(label_map).fillna(journey["step1"])

    l1 = journey.groupby(["step1_lbl", "step2"]).size().reset_index(name="val")
    l1.columns = ["src", "tgt", "val"]
    l2 = journey.groupby(["step2", "step3"]).size().reset_index(name="val")
    l2.columns = ["src", "tgt", "val"]

    links = pd.concat([l1, l2], ignore_index=True)
    links = links[links["val"] > 0]

    nodes = list(pd.concat([links["src"], links["tgt"]]).unique())
    node_map = {n: i for i, n in enumerate(nodes)}

    colors = []
    for n in nodes:
        if "High Intent" in n:
            colors.append("#10B981")
        elif "Bounced" in n:
            colors.append("#EF4444")
        elif "Kratix" in n:
            colors.append("#8B5CF6")
        elif "Syntasso" in n:
            colors.append("#3B82F6")
        else:
            colors.append("#9CA3AF")

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                    color=colors,
                    hovertemplate="<b>%{label}</b><extra></extra>",
                ),
                link=dict(
                    source=links["src"].map(node_map),
                    target=links["tgt"].map(node_map),
                    value=links["val"],
                    color="rgba(200, 200, 200, 0.3)",
                    hovertemplate="<b>From:</b> %{source.label}<br><b>To:</b> %{target.label}<br><b>Transitions:</b> %{value}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(height=600, margin=dict(t=20, b=20, l=10, r=10))
    return fig


def plot_dropoff_fancy(df: pd.DataFrame) -> go.Figure:
    bounces = df.groupby("uniq_sess").filter(lambda x: len(x) == 1)
    if bounces.empty:
        return go.Figure()

    col = "h1" if "h1" in bounces.columns else "page_name"
    data = bounces[col].astype(str).value_counts().head(8).reset_index()
    data.columns = ["Page", "Users"]

    fig = px.bar(data, x="Users", y="Page", orientation="h", text="Users", color="Users", color_continuous_scale="Reds")
    fig.update_layout(
        title="<b>Top Drop-off Pages</b>",
        plot_bgcolor="white",
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
        height=320,
    )
    fig.update_traces(textposition="outside")
    return fig


def page_intelligence_table_html(df: pd.DataFrame) -> str:
    if "url" not in df.columns or "uniq_sess" not in df.columns:
        return "<div style='padding:10px;color:#E5E7EB;'>Missing required columns (url / uniq_sess).</div>"

    d = df.sort_values(["uniq_sess", "activity_date"]).copy()
    landing = d.groupby("uniq_sess", as_index=False).first()

    landing["Top Landing Page (URL)"] = landing["url"].astype(str)

    # Unique Companies (prefer company_domain if present)
    company_col = "company_domain" if "company_domain" in landing.columns else ("company_name" if "company_name" in landing.columns else None)
    if company_col:
        companies = landing.groupby("Top Landing Page (URL)")[company_col].nunique().reset_index(name="Unique Companies")
    else:
        companies = landing.groupby("Top Landing Page (URL)").size().reset_index(name="Unique Companies")
        companies["Unique Companies"] = 0

    anon = landing.groupby("Top Landing Page (URL)")["uid"].nunique().reset_index(name="Anonymous Developers")

    dean_col = "deanon_id" if "deanon_id" in landing.columns else ("reo_dev_id" if "reo_dev_id" in landing.columns else None)
    if dean_col:
        deanon = (
            landing[landing["is_deanonymized"] == True]
            .groupby("Top Landing Page (URL)")[dean_col]
            .nunique()
            .reset_index(name="Deanonymized Developers")
        )
    else:
        deanon = landing.groupby("Top Landing Page (URL)").size().reset_index(name="Deanonymized Developers")
        deanon["Deanonymized Developers"] = 0

    agg = companies.merge(deanon, on="Top Landing Page (URL)", how="outer").merge(anon, on="Top Landing Page (URL)", how="outer").fillna(0)

    agg["Unique Companies"] = agg["Unique Companies"].astype(int)
    agg["Deanonymized Developers"] = agg["Deanonymized Developers"].astype(int)
    agg["Anonymous Developers"] = agg["Anonymous Developers"].astype(int)

    agg = agg.sort_values("Anonymous Developers", ascending=False).head(20)

    table_html = agg.to_html(index=False, classes="styled-table", border=0, escape=True)
    return f"<div class='table-container'>{table_html}</div>"


# =========================
# LOAD DATA
# =========================
try:
    df_all = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(f"CSV not found: '{DATA_FILE}'. Make sure the file exists in the repo root and the name matches exactly.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()


# =========================
# FILTER UI
# =========================
st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
st.markdown("<div class='filter-title'>🔍 Data Filters</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

# --- Date filter (UPDATED: dropdown + start/end, no range-picker UI) ---
with c1:
    if "activity_date" in df_all.columns:
        _d = df_all["activity_date"].dropna()
        if not _d.empty:
            _min_d = _d.min().date()
            _max_d = _d.max().date()
        else:
            _min_d = None
            _max_d = None
    else:
        _min_d = None
        _max_d = None

    date_mode = st.selectbox(
        "Select date range",
        options=["Auto date range", "Last 7 days", "Last 30 days", "This month", "This year", "Custom"],
        index=0,
    )

    start_date = None
    end_date = None

    if _min_d is not None and _max_d is not None:
        if date_mode == "Auto date range":
            start_date, end_date = _min_d, _max_d
        elif date_mode == "Last 7 days":
            end_date = _max_d
            start_date = max(_min_d, end_date - pd.Timedelta(days=6))
        elif date_mode == "Last 30 days":
            end_date = _max_d
            start_date = max(_min_d, end_date - pd.Timedelta(days=29))
        elif date_mode == "This month":
            end_date = _max_d
            start_date = end_date.replace(day=1)
            if start_date < _min_d:
                start_date = _min_d
        elif date_mode == "This year":
            end_date = _max_d
            start_date = end_date.replace(month=1, day=1)
            if start_date < _min_d:
                start_date = _min_d
        else:  # Custom
            c1a, c1b = st.columns(2)
            with c1a:
                start_date = st.date_input("Start date", value=_min_d, min_value=_min_d, max_value=_max_d)
            with c1b:
                end_date = st.date_input("End date", value=_max_d, min_value=_min_d, max_value=_max_d)
    else:
        # No valid dates in data; keep unset (no filtering applied)
        start_date, end_date = None, None

with c2:
    countries = sorted(df_all["org_hq_country"].dropna().unique()) if "org_hq_country" in df_all.columns else []
    sel_countries = st.multiselect("Country", options=countries)

with c3:
    industries = sorted(df_all["industry"].dropna().unique()) if "industry" in df_all.columns else []
    sel_industries = st.multiselect("Industry", options=industries)

# --- Employee size filter (FIXED: resolve overlaps) ---
with c4:
    size_candidates = [
        "normalised_employee_range",
        "employee_range",
        "employee_size",
        "company_size",
        "org_employee_range",
        "employee_count_range",
    ]
    size_cols = [c for c in size_candidates if c in df_all.columns]

    if size_cols:
        # pick first non-null value across overlapping fields
        df_all["_emp_size"] = df_all[size_cols].bfill(axis=1).iloc[:, 0]
        sizes = sorted(df_all["_emp_size"].dropna().unique())
    else:
        df_all["_emp_size"] = pd.NA
        sizes = []

    sel_sizes = st.multiselect("Size", options=sizes)

st.markdown("</div>", unsafe_allow_html=True)

# Apply filters
t = df_all.copy()

# Date filter (UPDATED)
if start_date is not None and end_date is not None and "activity_date" in t.columns:
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    t = t[t["activity_date"].notna()]
    t = t[(t["activity_date"] >= start_ts) & (t["activity_date"] <= end_ts)]

if sel_countries and "org_hq_country" in t.columns:
    t = t[t["org_hq_country"].isin(sel_countries)]
if sel_industries and "industry" in t.columns:
    t = t[t["industry"].isin(sel_industries)]
if sel_sizes and "_emp_size" in t.columns:
    t = t[t["_emp_size"].isin(sel_sizes)]

if t.empty:
    st.warning("No data matches the selected filters.")
    st.stop()


# =========================
# KPIs
# =========================
m1 = t[t["is_deanonymized"]]["deanon_id"].nunique() if "deanon_id" in t.columns else 0
m2 = t["company_name"].nunique() if "company_name" in t.columns else 0
m3 = t[t["activity_type"].isin(["CLI", "COPY_COMMAND", "FORM_FILLED"])].shape[0] if "activity_type" in t.columns else 0

st.markdown(
    f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-val">{m1}</div>
    <div class="metric-lbl">Deanonymized Developers</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">{m2}</div>
    <div class="metric-lbl">Unique Companies</div>
  </div>
  <div class="metric-card" style="border-left-color: #10B981">
    <div class="metric-val" style="color:#10B981">{m3}</div>
    <div class="metric-lbl">High Intent Actions</div>
  </div>
</div>
""",
    unsafe_allow_html=True
)


# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["Developer Journey", "Page Intelligence", "Raw Data"])

with tab1:
    st.plotly_chart(plot_sankey_detailed(t), use_container_width=True)
    st.plotly_chart(plot_dropoff_fancy(t), use_container_width=True)

with tab2:
    st.markdown(page_intelligence_table_html(t), unsafe_allow_html=True)

with tab3:
    cols = [c for c in DISPLAY_COLS if c in t.columns]
    if cols:
        st.dataframe(t[cols].head(250), use_container_width=True)
    else:
        st.dataframe(t.head(250), use_container_width=True)
