
import os
import io
import pandas as pd
import streamlit as st
import altair as alt

# ---------- Page config ----------
st.set_page_config(
    page_title="Streamlit + Altair Demo",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    # Ensure categorical ordering for nicer charts
    df["category"] = pd.Categorical(df["category"], categories=sorted(df["category"].unique()), ordered=True)
    df["region"] = pd.Categorical(df["region"], categories=sorted(df["region"].unique()), ordered=True)
    return df

DATA_PATH = os.path.join(os.path.dirname(__file__), "raw_data/sales_demo.csv")
df = load_data(DATA_PATH)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

cats = st.sidebar.multiselect("Category", options=list(df["category"].cat.categories), default=list(df["category"].cat.categories))
regs = st.sidebar.multiselect("Region", options=list(df["region"].cat.categories), default=list(df["region"].cat.categories))

# ---------- Filtered data ----------
mask = (
    (df["date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
    & (df["category"].isin(cats))
    & (df["region"].isin(regs))
)
fdf = df.loc[mask].copy()

# ---------- KPIs ----------
col1, col2, col3 = st.columns(3)
col1.metric("Total revenue", f"${fdf['revenue'].sum():,.0f}")
col2.metric("Total units", f"{fdf['units'].sum():,.0f}")
col3.metric("Average price", f"${fdf['price'].mean():.2f}")

st.markdown("---")

# ---------- Altair charts ----------
# Global selection for crossfiltering
selection = alt.selection_point(fields=["category", "region"], bind="legend")

# Line chart over time
line = (
    alt.Chart(fdf)
    .mark_line(point=True)
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("sum(revenue):Q", title="Revenue"),
        color=alt.Color("category:N", title="Category"),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("sum(revenue):Q", title="Revenue", format=",.0f")]
    )
    .add_params(selection)
    .transform_filter(selection)
    .properties(title="Revenue over time")
)

# Bar by category
bar_cat = (
    alt.Chart(fdf)
    .mark_bar()
    .encode(
        x=alt.X("sum(revenue):Q", title="Revenue"),
        y=alt.Y("category:N", sort="-x", title="Category"),
        color=alt.Color("category:N", legend=None),
        tooltip=[alt.Tooltip("sum(revenue):Q", title="Revenue", format=",.0f")]
    )
    .add_params(selection)
    .transform_filter(selection)
    .properties(title="Revenue by category")
)

# Bar by region
bar_reg = (
    alt.Chart(fdf)
    .mark_bar()
    .encode(
        x=alt.X("sum(revenue):Q", title="Revenue"),
        y=alt.Y("region:N", sort="-x", title="Region"),
        color=alt.Color("region:N", legend=None),
        tooltip=[alt.Tooltip("sum(revenue):Q", title="Revenue", format=",.0f")]
    )
    .add_params(selection)
    .transform_filter(selection)
    .properties(title="Revenue by region")
)

# Layout
upper = line.properties(height=320).resolve_scale(color="independent")
lower = alt.hconcat(bar_cat, bar_reg).resolve_scale(color="independent")

st.altair_chart(upper, use_container_width=True)
st.altair_chart(lower, use_container_width=True)

st.markdown("---")

# ---------- Data table and download ----------
with st.expander("Show filtered data"):
    st.dataframe(fdf, use_container_width=True, height=300)

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

st.download_button(
    label="Download filtered data (CSV)",
    data=to_csv_bytes(fdf),
    file_name="filtered_data.csv",
    mime="text/csv"
)

st.caption("Built with Streamlit + Altair · Demo app for teaching analytics.")
