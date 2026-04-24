"""
Manchester United (MANU) — Stock Price & Match Performance Analysis
ACC102 Mini Assignment – Track 4: Interactive Data Analysis Tool

Streamlit application that replicates the full analysis pipeline from the
notebook, loading pre-cleaned data from Excel files and presenting all
EDA, feature engineering, and statistical results interactively.

🔴 MANCHESTER UNITED THEMED VERSION 🔴
Background: Red, Yellow, White
Enhanced UI with interactive elements and team branding
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔴 MANU Stock & Match Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colours (Manchester United Theme) ───────────────────────────────────
MANU_RED   = "#DA291C"      # Primary red
MANU_WHITE = "#FFFFFF"      # White
MANU_YELLOW = "#FDB913"     # Gold/Yellow
MANU_BLACK = "#1a0000"      # Dark accent
RESULT_COLORS = {"Win": "#2ecc71", "Draw": "#95a5a6", "Loss": "#e74c3c"}

# ── Custom CSS for Manchester United Theme ─────────────────────────────────
st.markdown("""
<style>
    /* Main background - Red and White stripes effect */
    .main {
        background: linear-gradient(135deg, #DA291C 0%, #FFFFFF 50%, #FDB913 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #DA291C 0%, #1a0000 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Header styling */
    h1, h2 {
        color: #DA291C;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    h3 {
        color: #1a0000;
        border-left: 4px solid #FDB913;
        padding-left: 10px;
    }
    
    /* Metric cards styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(218, 41, 28, 0.1) 0%, rgba(253, 185, 19, 0.1) 100%);
        border: 2px solid #DA291C;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Tab styling */
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #DA291C;
        border-bottom: 3px solid #FDB913;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #DA291C 0%, #FDB913 100%);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #1a0000 0%, #DA291C 100%);
    }
    
    /* Dataframe styling */
    [data-testid="dataframe"] {
        border: 2px solid #DA291C;
        border-radius: 5px;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background-color: rgba(46, 204, 113, 0.2);
        border-left: 4px solid #2ecc71;
    }
    
    .stInfo {
        background-color: rgba(218, 41, 28, 0.1);
        border-left: 4px solid #DA291C;
    }
    
    /* Divider styling */
    hr {
        border: 2px solid #FDB913;
    }
</style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")
plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11})


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Load and merge stock + match data, then engineer all features."""

    # ── Stock data ────────────────────────────────────────────────────────────
    stock_raw = pd.read_excel(
        "MANU_stock_WRDS_CRSP.xlsx", sheet_name="CRSP_Monthly_Stock"
    )
    stock_raw["date"] = pd.to_datetime(stock_raw["date"])
    stock_df = stock_raw.rename(columns={
        "date":    "Month",
        "prc":     "price",
        "ret_adj": "monthly_return",
        "vol":     "volume",
    })
    stock_df["Month"] = stock_df["Month"].dt.to_period("M").dt.to_timestamp()

    # ── Match monthly aggregation ─────────────────────────────────────────────
    monthly_agg = pd.read_excel(
        "MANU_match_football_data.xlsx", sheet_name="Monthly_Match_Data"
    )
    monthly_agg["Month"] = (
        pd.to_datetime(monthly_agg["Month"]).dt.to_period("M").dt.to_timestamp()
    )

    # ── Match-level raw data ──────────────────────────────────────────────────
    match_raw = pd.read_excel(
        "MANU_match_football_data.xlsx", sheet_name="Match_Level_Raw"
    )
    match_raw["Date"]  = pd.to_datetime(match_raw["Date"])
    match_raw["Month"] = pd.to_datetime(match_raw["Month"])

    # ── Inner join ────────────────────────────────────────────────────────────
    merged = pd.merge(stock_df, monthly_agg, on="Month", how="inner")
    merged = merged.sort_values("Month").reset_index(drop=True)

    # ── Feature engineering (mirrors notebook Section 7.3) ───────────────────
    merged["Monthly_Return_Pct"] = (merged["monthly_return"] * 100).round(2)
    merged["Win_Rate"]           = (merged["Wins"] / merged["Matches"] * 100).round(1)
    merged["Avg_Goal_Diff"]      = (merged["Goal_Diff"] / merged["Matches"]).round(2)
    merged["Points_Per_Match"]   = (merged["Points"] / merged["Matches"]).round(2)

    merged["Return_3M_Avg"]  = merged["Monthly_Return_Pct"].rolling(3, min_periods=1).mean().round(2)
    merged["WinRate_3M_Avg"] = merged["Win_Rate"].rolling(3, min_periods=1).mean().round(1)

    def assign_season(m):
        y, mo = m.year, m.month
        return f"{y}/{y+1-2000:02d}" if mo >= 8 else f"{y-1}/{y-2000:02d}"

    merged["Season"] = merged["Month"].apply(assign_season)

    merged["Price_Change"]       = merged["price"].diff().round(4)
    merged["Stock_Trend"]        = merged["Monthly_Return_Pct"].apply(
        lambda x: "Up" if x > 0 else ("Flat" if x == 0 else "Down")
    )
    merged["Goals_Per_Match"]    = (merged["Goals_For"] / merged["Matches"]).round(2)
    merged["Conceded_Per_Match"] = (merged["Goals_Against"] / merged["Matches"]).round(2)
    merged["Momentum_Score"]     = (
        merged["Win_Rate"] * 0.4
        + merged["Goal_Diff"] * 1.5
        + merged["Form_Last5"] * 0.5
    ).round(2)

    def grade(ppm):
        if ppm >= 2.5:   return "A"
        elif ppm >= 2.0: return "B"
        elif ppm >= 1.5: return "C"
        else:            return "D"

    merged["Performance_Grade"]     = merged["Points_Per_Match"].apply(grade)
    merged["Cumulative_Return_Pct"] = (
        (1 + merged["monthly_return"]).cumprod() - 1
    ).mul(100).round(2)
    season_avg = merged.groupby("Season")["Monthly_Return_Pct"].transform("mean")
    merged["Return_vs_Season_Avg"]  = (merged["Monthly_Return_Pct"] - season_avg).round(2)

    return stock_df, monthly_agg, match_raw, merged


stock_df, monthly_agg, match_raw, merged = load_data()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILTERS & BRANDING
# ══════════════════════════════════════════════════════════════════════════════

# 曼联队徽和品牌信息
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 20px;">
    <h2 style="color: #FDB913; margin: 0;">🔴 MANCHESTER UNITED 🔴</h2>
    <p style="color: white; margin: 10px 0; font-size: 14px;">Stock & Match Performance Analysis</p>
    <p style="color: #FDB913; margin: 5px 0; font-size: 12px;">⚽ Premier League | 2019/20 - 2024/25</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    width=100,
)

st.sidebar.markdown("---")

# 季节过滤器
all_seasons = sorted(merged["Season"].unique())
selected_seasons = st.sidebar.multiselect(
    "🏆 Filter by Season",
    options=all_seasons,
    default=all_seasons,
)

if selected_seasons:
    df = merged[merged["Season"].isin(selected_seasons)].copy()
    mr = match_raw[match_raw["Season"].isin(selected_seasons)].copy()
else:
    df = merged.copy()
    mr = match_raw.copy()

# 性能指标过滤器
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Performance Filters")
min_win_rate = st.sidebar.slider("Minimum Win Rate (%)", 0.0, 100.0, 0.0)
df = df[df["Win_Rate"] >= min_win_rate]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**📚 Data Sources**\n\n"
    "- Stock: WRDS / CRSP (`crsp.msf`)\n"
    "- Matches: [football-data.co.uk](https://www.football-data.co.uk)\n\n"
    "**🎯 Period:** Aug 2019 – Dec 2024"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='text-align: center; color: #FDB913; font-size: 12px;'>"
    "🔴 Red • Yellow • White 🔴<br>"
    "Manchester United Theme</p>",
    unsafe_allow_html=True
)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER WITH BRANDING
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #DA291C 0%, #FDB913 50%, #FFFFFF 100%); border-radius: 10px; margin-bottom: 20px;">
    <h1 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); margin: 0;">
        ⚽ Manchester United — Stock & Match Performance Analysis ⚽
    </h1>
    <p style="color: #1a0000; font-size: 16px; margin: 10px 0;">
        Interactive exploration of MANU stock price movements alongside Premier League match results
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI metrics row with enhanced styling ──────────────────────────────────────
st.markdown("### 📈 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "📅 Months Analysed",
        len(df),
        delta=None,
        delta_color="normal"
    )

with col2:
    st.metric(
        "⚽ Total Matches",
        mr["Date"].nunique() if len(mr) else 0,
        delta=None,
        delta_color="normal"
    )

with col3:
    win_rate = df['Win_Rate'].mean()
    st.metric(
        "🏆 Overall Win Rate",
        f"{win_rate:.1f}%",
        delta=f"{win_rate - 50:.1f}%" if win_rate > 50 else f"{win_rate - 50:.1f}%",
        delta_color="inverse" if win_rate > 50 else "off"
    )

with col4:
    avg_return = df['Monthly_Return_Pct'].mean()
    st.metric(
        "💰 Avg Monthly Return",
        f"{avg_return:.2f}%",
        delta=f"{avg_return:.2f}%",
        delta_color="normal" if avg_return > 0 else "inverse"
    )

with col5:
    cum_return = df['Cumulative_Return_Pct'].iloc[-1] if len(df) else 0
    st.metric(
        "📊 Cumulative Return",
        f"{cum_return:.1f}%",
        delta=f"{cum_return:.1f}%",
        delta_color="normal" if cum_return > 0 else "inverse"
    )

st.markdown("---")

# 性能等级分布（新增交互元素）
st.markdown("### 🎯 Performance Grade Summary")
grade_dist = df["Performance_Grade"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
grade_colors_map = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}

grade_cols = st.columns(4)
for idx, (grade, count) in enumerate(grade_dist.items()):
    with grade_cols[idx]:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #DA291C 0%, #FDB913 100%); 
                    padding: 15px; border-radius: 10px; text-align: center; color: white;">
            <h3 style="margin: 0; color: white;">Grade {grade}</h3>
            <p style="font-size: 24px; margin: 10px 0; color: white;"><b>{int(count)}</b> months</p>
            <p style="margin: 0; font-size: 12px;">
                {grade_colors_map[grade]} 
                {'Excellent' if grade == 'A' else 'Good' if grade == 'B' else 'Fair' if grade == 'C' else 'Poor'}
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Stock Overview",
    "⚽ Match Performance",
    "🔗 Stock vs Match",
    "📊 EDA & Distributions",
    "📐 Statistical Tests",
    "🗃️ Raw Data",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STOCK OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("📊 Stock Price Timeline")

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df["Month"], df["price"], color=MANU_RED, lw=2.5, label="Closing Price (USD)")

    # Colour scatter by dominant monthly result
    for result, colour in RESULT_COLORS.items():
        mask = df["Result"] == result
        ax.scatter(df.loc[mask, "Month"], df.loc[mask, "price"],
                   color=colour, s=80, zorder=5, label=result, alpha=0.7)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlabel("Month", fontweight="bold")
    ax.set_ylabel("Price (USD)", fontweight="bold")
    ax.set_title("MANU Monthly Closing Price — Coloured by Dominant Monthly Result", fontweight="bold", color=MANU_RED)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Monthly return bar chart ───────────────────────────────────────────────
    st.subheader("💹 Monthly Return (%)")
    fig2, ax2 = plt.subplots(figsize=(13, 3.5))
    colors = [MANU_RED if r >= 0 else MANU_BLACK for r in df["Monthly_Return_Pct"]]
    ax2.bar(df["Month"], df["Monthly_Return_Pct"], color=colors, width=20, alpha=0.8)
    ax2.axhline(0, color="grey", lw=0.8, ls="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax2.set_ylabel("Return (%)", fontweight="bold")
    ax2.set_title("MANU Monthly Return (%)", fontweight="bold", color=MANU_RED)
    ax2.grid(True, alpha=0.3, axis='y')
    fig2.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Cumulative return ─────────────────────────────────────────────────────
    st.subheader("📈 Cumulative Return (%)")
    fig3, ax3 = plt.subplots(figsize=(13, 3.5))
    ax3.plot(df["Month"], df["Cumulative_Return_Pct"], color=MANU_RED, lw=2.5)
    ax3.fill_between(df["Month"], df["Cumulative_Return_Pct"], 0,
                     where=(df["Cumulative_Return_Pct"] >= 0),
                     alpha=0.25, color=MANU_YELLOW, label="Positive")
    ax3.fill_between(df["Month"], df["Cumulative_Return_Pct"], 0,
                     where=(df["Cumulative_Return_Pct"] < 0),
                     alpha=0.25, color=MANU_BLACK, label="Negative")
    ax3.axhline(0, color="grey", lw=0.8, ls="--")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax3.set_ylabel("Cumulative Return (%)", fontweight="bold")
    ax3.set_title("MANU Cumulative Return (%) — From Start of Selected Period", fontweight="bold", color=MANU_RED)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    fig3.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Season-level summary table ────────────────────────────────────────────
    st.subheader("🏆 Season-Level Stock Summary")
    season_stock = df.groupby("Season").agg(
        Avg_Price       = ("price",              "mean"),
        Avg_Return_Pct  = ("Monthly_Return_Pct", "mean"),
        Volatility      = ("Monthly_Return_Pct", "std"),
        Avg_Volume      = ("volume",             "mean"),
        Avg_Mktcap_M    = ("mktcap",             lambda x: (x.mean() / 1000).round(1)),
    ).round(2).reset_index()
    season_stock.columns = [
        "Season", "Avg Price (USD)", "Avg Return (%)", "Volatility (%)", "Avg Volume", "Avg Mktcap (B)"
    ]
    st.dataframe(season_stock, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MATCH PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("⚽ Match Results Timeline")

    fig, ax = plt.subplots(figsize=(13, 4))
    for result, colour in RESULT_COLORS.items():
        mask = df["Result"] == result
        ax.scatter(df.loc[mask, "Month"], df.loc[mask, "Win_Rate"],
                   color=colour, s=100, alpha=0.7, label=result, edgecolors="black", linewidth=1)

    ax.plot(df["Month"], df["Win_Rate"], color=MANU_RED, lw=1.5, alpha=0.3, linestyle="--")
    ax.axhline(50, color="grey", lw=1, ls="--", alpha=0.5, label="50% Win Rate")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlabel("Month", fontweight="bold")
    ax.set_ylabel("Win Rate (%)", fontweight="bold")
    ax.set_title("MANU Monthly Win Rate — Coloured by Dominant Result", fontweight="bold", color=MANU_RED)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Goals analysis ────────────────────────────────────────────────────────
    st.subheader("⚽ Goals For vs Against")
    col_gf, col_ga = st.columns(2)

    with col_gf:
        fig_gf, ax_gf = plt.subplots(figsize=(6, 3.5))
        ax_gf.bar(df["Month"], df["Goals_For"], color=MANU_RED, alpha=0.7, label="Goals For")
        ax_gf.set_ylabel("Goals", fontweight="bold")
        ax_gf.set_title("Goals Scored Per Month", fontweight="bold", color=MANU_RED)
        ax_gf.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_gf.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45, ha="right", fontsize=8)
        ax_gf.grid(True, alpha=0.3, axis='y')
        fig_gf.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig_gf)
        plt.close(fig_gf)

    with col_ga:
        fig_ga, ax_ga = plt.subplots(figsize=(6, 3.5))
        ax_ga.bar(df["Month"], df["Goals_Against"], color=MANU_BLACK, alpha=0.7, label="Goals Against")
        ax_ga.set_ylabel("Goals", fontweight="bold")
        ax_ga.set_title("Goals Conceded Per Month", fontweight="bold", color=MANU_RED)
        ax_ga.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_ga.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45, ha="right", fontsize=8)
        ax_ga.grid(True, alpha=0.3, axis='y')
        fig_ga.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig_ga)
        plt.close(fig_ga)

    # ── Season match summary ──────────────────────────────────────────────────
    st.subheader("🏆 Season-Level Match Summary")
    season_match = df.groupby("Season").agg(
        Matches     = ("Matches", "sum"),
        Wins        = ("Wins",    "sum"),
        Draws       = ("Draws",   "sum"),
        Losses      = ("Losses",  "sum"),
        Goals_For   = ("Goals_For",     "sum"),
        Goals_Against = ("Goals_Against", "sum"),
        Win_Rate    = ("Win_Rate", "mean"),
        Avg_Points  = ("Points_Per_Match", "mean"),
    ).round(2).reset_index()
    st.dataframe(season_match, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STOCK VS MATCH
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("🔗 Stock Price vs Win Rate Correlation")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df["Win_Rate"], df["price"], 
                        c=df["Monthly_Return_Pct"], cmap="RdYlGn",
                        s=100, alpha=0.6, edgecolors="black", linewidth=1)
    ax.set_xlabel("Win Rate (%)", fontweight="bold")
    ax.set_ylabel("Stock Price (USD)", fontweight="bold")
    ax.set_title("Stock Price vs Match Win Rate", fontweight="bold", color=MANU_RED)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Monthly Return (%)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Correlation with return ───────────────────────────────────────────────
    st.subheader("📊 Match Metrics vs Stock Return")
    corr_cols = ["Win_Rate", "Goal_Diff", "Points_Per_Match", "Momentum_Score", "Goals_Per_Match"]
    corr_data = []
    for col in corr_cols:
        valid = df[["Monthly_Return_Pct", col]].dropna()
        if len(valid) > 2:
            r, p = stats.pearsonr(valid["Monthly_Return_Pct"], valid[col])
            corr_data.append({
                "Metric": col.replace("_", " "),
                "Correlation": round(r, 4),
                "P-Value": round(p, 4),
                "Significant": "✅" if p < 0.05 else "❌"
            })
    corr_df = pd.DataFrame(corr_data)
    st.dataframe(corr_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EDA & DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("📊 Exploratory Data Analysis")

    col_hist1, col_hist2 = st.columns(2)

    with col_hist1:
        st.subheader("Return Distribution")
        fig1, ax1 = plt.subplots(figsize=(5.5, 4))
        ax1.hist(df["Monthly_Return_Pct"], bins=15, color=MANU_RED, alpha=0.7, edgecolor="white")
        ax1.axvline(df["Monthly_Return_Pct"].mean(), color=MANU_YELLOW, lw=2, linestyle="--",
                   label=f"Mean={df['Monthly_Return_Pct'].mean():.2f}%")
        ax1.set_xlabel("Monthly Return (%)", fontweight="bold")
        ax1.set_ylabel("Frequency", fontweight="bold")
        ax1.set_title("Distribution of Monthly Returns", fontweight="bold", color=MANU_RED)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        fig1.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col_hist2:
        st.subheader("Return by Result")
        fig2, ax2 = plt.subplots(figsize=(5.5, 4))
        for result, colour in RESULT_COLORS.items():
            data = df[df["Result"] == result]["Monthly_Return_Pct"].dropna()
            if len(data) > 1:
                ax2.hist(data, bins=10, alpha=0.6, color=colour,
                         edgecolor="white", label=result)
        ax2.set_xlabel("Monthly Return (%)", fontweight="bold")
        ax2.set_ylabel("Frequency", fontweight="bold")
        ax2.set_title("Return Distribution by Result", fontweight="bold", color=MANU_RED)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        fig2.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("🔥 Correlation Heatmap")
    corr_cols = [
        "Monthly_Return_Pct", "Win_Rate", "Goal_Diff",
        "Points_Per_Match", "Momentum_Score",
        "Goals_Per_Match", "Conceded_Per_Match",
    ]
    corr = df[corr_cols].corr()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, vmin=-1, vmax=1, ax=ax3,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax3.set_title("Correlation Matrix — Stock & Match Metrics", fontweight="bold", color=MANU_RED)
    fig3.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Box plot: return by result ─────────────────────────────────────────────
    st.subheader("📦 Monthly Return by Result (Box Plot)")
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    data_by_result = [
        df[df["Result"] == r]["Monthly_Return_Pct"].dropna().values
        for r in ["Win", "Draw", "Loss"]
    ]
    bp = ax4.boxplot(
        data_by_result,
        labels=["Win", "Draw", "Loss"],
        patch_artist=True,
        medianprops={"color": "black", "lw": 2},
    )
    for patch, colour in zip(bp["boxes"], [RESULT_COLORS["Win"],
                                            RESULT_COLORS["Draw"],
                                            RESULT_COLORS["Loss"]]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    ax4.axhline(0, color="grey", lw=0.8, ls="--")
    ax4.set_ylabel("Monthly Return (%)", fontweight="bold")
    ax4.set_title("Monthly Return Distribution by Result", fontweight="bold", color=MANU_RED)
    ax4.grid(True, alpha=0.3, axis='y')
    fig4.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    # ── Performance grade distribution ────────────────────────────────────────
    st.subheader("🎯 Performance Grade Distribution")
    grade_counts = df["Performance_Grade"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    grade_colors = {"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c"}
    fig5, ax5 = plt.subplots(figsize=(5, 3.5))
    ax5.bar(grade_counts.index,
            grade_counts.values,
            color=[grade_colors[g] for g in grade_counts.index],
            alpha=0.8, edgecolor="black", linewidth=1.5)
    ax5.set_xlabel("Performance Grade", fontweight="bold")
    ax5.set_ylabel("Number of Months", fontweight="bold")
    ax5.set_title("Performance Grade Distribution\n(A≥2.5, B≥2.0, C≥1.5, D<1.5 pts/match)", 
                 fontweight="bold", color=MANU_RED)
    ax5.grid(True, alpha=0.3, axis='y')
    fig5.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    # ── Momentum score timeline ───────────────────────────────────────────────
    st.subheader("🚀 Momentum Score Over Time")
    fig6, ax6 = plt.subplots(figsize=(13, 3.5))
    ax6.plot(df["Month"], df["Momentum_Score"], color=MANU_RED, lw=2.5)
    ax6.fill_between(df["Month"], df["Momentum_Score"],
                     df["Momentum_Score"].mean(),
                     where=(df["Momentum_Score"] >= df["Momentum_Score"].mean()),
                     alpha=0.3, color=MANU_YELLOW)
    ax6.axhline(df["Momentum_Score"].mean(), color="grey", lw=1, ls="--",
                label=f"Mean={df['Momentum_Score'].mean():.1f}")
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax6.set_ylabel("Momentum Score", fontweight="bold")
    ax6.set_title("Monthly Momentum Score (Win Rate × 0.4 + Goal Diff × 1.5 + Form × 0.5)", 
                 fontweight="bold", color=MANU_RED)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    fig6.patch.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("📐 Statistical Analysis")

    # ── One-way ANOVA: return by result ───────────────────────────────────────
    st.markdown("#### 9.1  One-Way ANOVA — Monthly Return by Match Result")
    win_ret  = df[df["Result"] == "Win"]["Monthly_Return_Pct"].dropna()
    draw_ret = df[df["Result"] == "Draw"]["Monthly_Return_Pct"].dropna()
    loss_ret = df[df["Result"] == "Loss"]["Monthly_Return_Pct"].dropna()

    if len(win_ret) > 1 and len(draw_ret) > 1 and len(loss_ret) > 1:
        f_stat, p_anova = stats.f_oneway(win_ret, draw_ret, loss_ret)
        anova_df = pd.DataFrame({
            "Group":  ["Win",          "Draw",          "Loss"],
            "N":      [len(win_ret),   len(draw_ret),   len(loss_ret)],
            "Mean (%)": [win_ret.mean().round(3),
                         draw_ret.mean().round(3),
                         loss_ret.mean().round(3)],
            "Std (%)":  [win_ret.std().round(3),
                         draw_ret.std().round(3),
                         loss_ret.std().round(3)],
        })
        st.dataframe(anova_df, use_container_width=True, hide_index=True)
        st.markdown(f"**F-statistic = {f_stat:.4f},  p-value = {p_anova:.4f}**")
        if p_anova < 0.05:
            st.success("✅ Significant difference in returns across result groups (p < 0.05).")
        else:
            st.info("ℹ️ No statistically significant difference in returns across result groups (p ≥ 0.05).")
    else:
        st.warning("Insufficient data for ANOVA with current season filter.")

    st.markdown("---")

    # ── Pearson correlations ──────────────────────────────────────────────────
    st.markdown("#### 9.2  Pearson Correlations with Monthly Return")
    match_metrics = [
        "Win_Rate", "Goal_Diff", "Points_Per_Match",
        "Momentum_Score", "Goals_Per_Match", "Conceded_Per_Match",
    ]
    corr_rows = []
    for col in match_metrics:
        valid = df[["Monthly_Return_Pct", col]].dropna()
        if len(valid) > 2:
            r, p = stats.pearsonr(valid["Monthly_Return_Pct"], valid[col])
            corr_rows.append({
                "Metric":      col.replace("_", " "),
                "Pearson r":   round(r, 4),
                "p-value":     round(p, 4),
                "Significant": "✅" if p < 0.05 else "—",
            })
    corr_table = pd.DataFrame(corr_rows)
    st.dataframe(corr_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── OLS regression ────────────────────────────────────────────────────────
    st.markdown("#### 9.3  Simple OLS Regression: Return ~ Win Rate")
    valid_ols = df[["Monthly_Return_Pct", "Win_Rate"]].dropna()
    if len(valid_ols) > 2:
        slope, intercept, r_val, p_val, se = stats.linregress(
            valid_ols["Win_Rate"], valid_ols["Monthly_Return_Pct"]
        )
        ols_df = pd.DataFrame({
            "Coefficient":  ["Intercept",     "Win Rate"],
            "Estimate":     [round(intercept, 4), round(slope, 4)],
            "p-value":      ["—",             round(p_val, 4)],
            "R²":           [round(r_val**2, 4), ""],
        })
        st.dataframe(ols_df, use_container_width=True, hide_index=True)
        st.markdown(
            f"**Equation:** Monthly Return = {intercept:.3f} + {slope:.4f} × Win Rate  "
            f"(R² = {r_val**2:.4f}, p = {p_val:.4f})"
        )

    st.markdown("---")

    # ── T-test: win months vs non-win months ──────────────────────────────────
    st.markdown("#### 9.4  Independent t-Test — Win Months vs Non-Win Months")
    win_months  = df[df["Result"] == "Win"]["Monthly_Return_Pct"].dropna()
    nonwin_months = df[df["Result"] != "Win"]["Monthly_Return_Pct"].dropna()
    if len(win_months) > 1 and len(nonwin_months) > 1:
        t_stat, p_ttest = stats.ttest_ind(win_months, nonwin_months, equal_var=False)
        ttest_df = pd.DataFrame({
            "Group":    ["Win months",    "Non-Win months"],
            "N":        [len(win_months), len(nonwin_months)],
            "Mean (%)": [win_months.mean().round(3), nonwin_months.mean().round(3)],
            "Std (%)":  [win_months.std().round(3),  nonwin_months.std().round(3)],
        })
        st.dataframe(ttest_df, use_container_width=True, hide_index=True)
        st.markdown(f"**t-statistic = {t_stat:.4f},  p-value = {p_ttest:.4f}** (Welch's t-test)")
        if p_ttest < 0.05:
            st.success("✅ Significant difference in returns between win and non-win months (p < 0.05).")
        else:
            st.info("ℹ️ No statistically significant difference between win and non-win months (p ≥ 0.05).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.subheader("📋 Merged Monthly Dataset")
    display_cols = [
        "Month", "Season", "price", "Monthly_Return_Pct", "Cumulative_Return_Pct",
        "Matches", "Wins", "Draws", "Losses", "Goals_For", "Goals_Against",
        "Goal_Diff", "Points", "Win_Rate", "Points_Per_Match",
        "Momentum_Score", "Performance_Grade", "Stock_Trend",
    ]
    st.dataframe(
        df[display_cols].rename(columns={"price": "Price (USD)"}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("⚽ Match-Level Raw Data")
    st.dataframe(mr, use_container_width=True, hide_index=True)

    st.subheader("📥 Download Data")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_merged = df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Merged Data (CSV)",
            csv_merged,
            "MANU_merged_monthly.csv",
            "text/csv",
        )
    with col_dl2:
        csv_match = mr.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Match-Level Data (CSV)",
            csv_match,
            "MANU_match_level.csv",
            "text/csv",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #DA291C 0%, #FDB913 50%, #FFFFFF 100%); border-radius: 10px;">
    <p style="color: #1a0000; margin: 5px 0; font-size: 12px; font-weight: bold;">
        🔴 MANCHESTER UNITED THEMED ANALYSIS 🔴
    </p>
    <p style="color: #1a0000; margin: 5px 0; font-size: 11px;">
        Data: WRDS / CRSP (stock) · football-data.co.uk (matches)<br>
        Period: Aug 2019 – Dec 2024 · ACC102 Mini Assignment Track 4<br>
        Theme: Red • Yellow • White | Enhanced UI & Interactive Elements
    </p>
</div>
""", unsafe_allow_html=True)
