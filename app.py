"""
Manchester United (MANU) — Stock Price & Match Performance Analysis
ACC102 Mini Assignment – Track 4: Interactive Data Analysis Tool

Streamlit application that replicates the full analysis pipeline from the
notebook, loading pre-cleaned data from Excel files and presenting all
EDA, feature engineering, and statistical results interactively.
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
    page_title="MANU Stock & Match Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colours ─────────────────────────────────────────────────────────────
MANU_RED   = "#DA291C"
MANU_BLACK = "#1a0000"
MANU_GOLD  = "#FFD700"
RESULT_COLORS = {"Win": "#2ecc71", "Draw": "#95a5a6", "Loss": "#e74c3c"}

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
# SIDEBAR — FILTERS
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
    width=80,
)
st.sidebar.title("MANU Analysis")
st.sidebar.markdown("**ACC102 Mini Assignment — Track 4**")
st.sidebar.markdown("---")

all_seasons = sorted(merged["Season"].unique())
selected_seasons = st.sidebar.multiselect(
    "Filter by Season",
    options=all_seasons,
    default=all_seasons,
)

if selected_seasons:
    df = merged[merged["Season"].isin(selected_seasons)].copy()
    mr = match_raw[match_raw["Season"].isin(selected_seasons)].copy()
else:
    df = merged.copy()
    mr = match_raw.copy()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data Sources**\n\n"
    "- Stock: WRDS / CRSP (`crsp.msf`)\n"
    "- Matches: [football-data.co.uk](https://www.football-data.co.uk)"
)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.title("⚽ Manchester United — Stock & Match Performance Analysis")
st.markdown(
    "Interactive exploration of MANU stock price movements alongside "
    "Premier League match results (2019/20 – 2024/25)."
)

# ── KPI metrics row ───────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Months Analysed",   len(df))
col2.metric("Total Matches",     mr["Date"].nunique() if len(mr) else 0)
col3.metric("Overall Win Rate",  f"{df['Win_Rate'].mean():.1f}%")
col4.metric("Avg Monthly Return",f"{df['Monthly_Return_Pct'].mean():.2f}%")
col5.metric("Cumulative Return", f"{df['Cumulative_Return_Pct'].iloc[-1]:.1f}%" if len(df) else "N/A")

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
    st.subheader("8.1  Stock Price Timeline")

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df["Month"], df["price"], color=MANU_RED, lw=2, label="Closing Price (USD)")

    # Colour scatter by dominant monthly result
    for result, colour in RESULT_COLORS.items():
        mask = df["Result"] == result
        ax.scatter(df.loc[mask, "Month"], df.loc[mask, "price"],
                   color=colour, s=60, zorder=5, label=result)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (USD)")
    ax.set_title("MANU Monthly Closing Price — coloured by dominant monthly result")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Monthly return bar chart ───────────────────────────────────────────────
    st.subheader("Monthly Return (%)")
    fig2, ax2 = plt.subplots(figsize=(13, 3.5))
    colors = [MANU_RED if r >= 0 else MANU_BLACK for r in df["Monthly_Return_Pct"]]
    ax2.bar(df["Month"], df["Monthly_Return_Pct"], color=colors, width=20)
    ax2.axhline(0, color="grey", lw=0.8, ls="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax2.set_ylabel("Return (%)")
    ax2.set_title("MANU Monthly Return (%)")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Cumulative return ─────────────────────────────────────────────────────
    st.subheader("Cumulative Return (%)")
    fig3, ax3 = plt.subplots(figsize=(13, 3.5))
    ax3.plot(df["Month"], df["Cumulative_Return_Pct"], color=MANU_GOLD, lw=2)
    ax3.fill_between(df["Month"], df["Cumulative_Return_Pct"], 0,
                     where=(df["Cumulative_Return_Pct"] >= 0),
                     alpha=0.25, color=MANU_RED, label="Positive")
    ax3.fill_between(df["Month"], df["Cumulative_Return_Pct"], 0,
                     where=(df["Cumulative_Return_Pct"] < 0),
                     alpha=0.25, color=MANU_BLACK, label="Negative")
    ax3.axhline(0, color="grey", lw=0.8, ls="--")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax3.set_ylabel("Cumulative Return (%)")
    ax3.set_title("MANU Cumulative Return (%) — from start of selected period")
    ax3.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Season-level summary table ────────────────────────────────────────────
    st.subheader("Season-Level Stock Summary")
    season_stock = df.groupby("Season").agg(
        Avg_Price       = ("price",              "mean"),
        Avg_Return_Pct  = ("Monthly_Return_Pct", "mean"),
        Volatility      = ("Monthly_Return_Pct", "std"),
        Avg_Volume      = ("volume",             "mean"),
        Avg_Mktcap_M    = ("mktcap",             lambda x: (x.mean() / 1000).round(1)),
    ).round(2).reset_index()
    season_stock.columns = [
        "Season", "Avg Price (USD)", "Avg Monthly Return (%)",
        "Volatility (Std)", "Avg Volume (000s)", "Avg Mkt Cap ($M)"
    ]
    st.dataframe(season_stock, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MATCH PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Match Result Distribution")

    col_a, col_b = st.columns(2)

    with col_a:
        result_counts = mr["Result"].value_counts().reindex(["Win", "Draw", "Loss"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            result_counts,
            labels=result_counts.index,
            colors=[RESULT_COLORS[r] for r in result_counts.index],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Overall Result Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        season_results = (
            mr.groupby(["Season", "Result"])
            .size()
            .unstack(fill_value=0)
            [["Win", "Draw", "Loss"]]
        )
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        season_results.plot(
            kind="bar", stacked=True, ax=ax2,
            color=[RESULT_COLORS["Win"], RESULT_COLORS["Draw"], RESULT_COLORS["Loss"]],
        )
        ax2.set_xlabel("Season")
        ax2.set_ylabel("Matches")
        ax2.set_title("Results by Season")
        plt.xticks(rotation=30, ha="right")
        ax2.legend(loc="upper right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Monthly points & goal diff ────────────────────────────────────────────
    st.subheader("Monthly Points & Goal Difference")
    fig3, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].bar(df["Month"], df["Points"], color=MANU_RED, width=20)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[0].set_ylabel("Points")
    axes[0].set_title("Monthly Points")

    bar_colors = [MANU_RED if g >= 0 else MANU_BLACK for g in df["Goal_Diff"]]
    axes[1].bar(df["Month"], df["Goal_Diff"], color=bar_colors, width=20)
    axes[1].axhline(0, color="grey", lw=0.8, ls="--")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    axes[1].set_ylabel("Goal Difference")
    axes[1].set_title("Monthly Goal Difference")

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Home vs Away breakdown ────────────────────────────────────────────────
    st.subheader("Home vs Away Performance")
    home_away = (
        mr.groupby(["IsHome", "Result"])
        .size()
        .unstack(fill_value=0)
        [["Win", "Draw", "Loss"]]
    )
    home_away.index = ["Away", "Home"]

    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    home_away.plot(
        kind="bar", ax=ax4,
        color=[RESULT_COLORS["Win"], RESULT_COLORS["Draw"], RESULT_COLORS["Loss"]],
    )
    ax4.set_xlabel("")
    ax4.set_ylabel("Matches")
    ax4.set_title("Home vs Away Result Breakdown")
    plt.xticks(rotation=0)
    ax4.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    # ── Season-level match summary ────────────────────────────────────────────
    st.subheader("Season-Level Match Summary")
    season_match = df.groupby("Season").agg(
        Matches        = ("Matches",        "sum"),
        Wins           = ("Wins",           "sum"),
        Draws          = ("Draws",          "sum"),
        Losses         = ("Losses",         "sum"),
        Goals_For      = ("Goals_For",      "sum"),
        Goals_Against  = ("Goals_Against",  "sum"),
        Total_Points   = ("Points",         "sum"),
        Avg_Win_Rate   = ("Win_Rate",       "mean"),
        Avg_Momentum   = ("Momentum_Score", "mean"),
    ).round(1).reset_index()
    st.dataframe(season_match, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — STOCK vs MATCH
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("8.2  Stock Return vs Match Performance (Scatter)")

    x_axis = st.selectbox(
        "X-axis (match metric)",
        ["Win_Rate", "Goal_Diff", "Points_Per_Match", "Momentum_Score",
         "Goals_Per_Match", "Conceded_Per_Match", "Form_Last5"],
        index=0,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for result, colour in RESULT_COLORS.items():
        mask = df["Result"] == result
        ax.scatter(
            df.loc[mask, x_axis],
            df.loc[mask, "Monthly_Return_Pct"],
            color=colour, label=result, alpha=0.75, s=60,
        )

    # Regression line
    valid = df[[x_axis, "Monthly_Return_Pct"]].dropna()
    if len(valid) > 2:
        slope, intercept, r, p, _ = stats.linregress(
            valid[x_axis], valid["Monthly_Return_Pct"]
        )
        x_line = np.linspace(valid[x_axis].min(), valid[x_axis].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", lw=1.5,
                label=f"OLS  r={r:.2f}  p={p:.3f}")

    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(x_axis.replace("_", " "))
    ax.set_ylabel("Monthly Return (%)")
    ax.set_title(f"Monthly Return vs {x_axis.replace('_', ' ')}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Dual-axis timeline ────────────────────────────────────────────────────
    st.subheader("Dual-Axis: Stock Return & Win Rate Over Time")
    fig2, ax1 = plt.subplots(figsize=(13, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(df["Month"], df["Monthly_Return_Pct"], color=MANU_RED,
             lw=2, label="Monthly Return (%)")
    ax1.fill_between(df["Month"], df["Monthly_Return_Pct"], 0,
                     alpha=0.15, color=MANU_RED)
    ax2.plot(df["Month"], df["Win_Rate"], color="#3498db",
             lw=2, ls="--", label="Win Rate (%)")

    ax1.set_ylabel("Monthly Return (%)", color=MANU_RED)
    ax2.set_ylabel("Win Rate (%)",        color="#3498db")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax1.set_title("Monthly Stock Return vs Win Rate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Season-level averages ─────────────────────────────────────────────────
    st.subheader("Season Averages: Return vs Win Rate")
    season_stats = df.groupby("Season").agg(
        Avg_Return  = ("Monthly_Return_Pct", "mean"),
        Avg_WinRate = ("Win_Rate",           "mean"),
        Avg_GoalDiff= ("Goal_Diff",          "mean"),
        Avg_Momentum= ("Momentum_Score",     "mean"),
    ).round(2).reset_index()

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(season_stats))
    w = 0.35
    ax3.bar(x - w/2, season_stats["Avg_Return"],  w, color=MANU_RED,   label="Avg Return (%)")
    ax3.bar(x + w/2, season_stats["Avg_WinRate"], w, color="#3498db",  label="Avg Win Rate (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(season_stats["Season"], rotation=30, ha="right")
    ax3.axhline(0, color="grey", lw=0.8, ls="--")
    ax3.set_ylabel("Value")
    ax3.set_title("Season Averages: Monthly Return vs Win Rate")
    ax3.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.dataframe(season_stats, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EDA & DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Return Distribution")

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(df["Monthly_Return_Pct"].dropna(), bins=15,
                color=MANU_RED, edgecolor="white", alpha=0.85)
        ax.axvline(df["Monthly_Return_Pct"].mean(), color=MANU_BLACK,
                   lw=2, ls="--", label=f"Mean={df['Monthly_Return_Pct'].mean():.2f}%")
        ax.set_xlabel("Monthly Return (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Monthly Returns")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(5.5, 4))
        for result, colour in RESULT_COLORS.items():
            data = df[df["Result"] == result]["Monthly_Return_Pct"].dropna()
            if len(data) > 1:
                ax2.hist(data, bins=10, alpha=0.6, color=colour,
                         edgecolor="white", label=result)
        ax2.set_xlabel("Monthly Return (%)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Return Distribution by Dominant Monthly Result")
        ax2.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("Correlation Matrix")
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
    ax3.set_title("Correlation Matrix — Stock & Match Metrics")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Box plot: return by result ─────────────────────────────────────────────
    st.subheader("Monthly Return by Dominant Result (Box Plot)")
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
    ax4.set_ylabel("Monthly Return (%)")
    ax4.set_title("Monthly Return Distribution by Dominant Monthly Result")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    # ── Performance grade distribution ────────────────────────────────────────
    st.subheader("Performance Grade Distribution")
    grade_counts = df["Performance_Grade"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    grade_colors = {"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c"}
    fig5, ax5 = plt.subplots(figsize=(5, 3.5))
    ax5.bar(grade_counts.index,
            grade_counts.values,
            color=[grade_colors[g] for g in grade_counts.index])
    ax5.set_xlabel("Performance Grade (Points per Match)")
    ax5.set_ylabel("Months")
    ax5.set_title("Monthly Performance Grade\n(A≥2.5, B≥2.0, C≥1.5, D<1.5 pts/match)")
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    # ── Momentum score timeline ───────────────────────────────────────────────
    st.subheader("Momentum Score Over Time")
    fig6, ax6 = plt.subplots(figsize=(13, 3.5))
    ax6.plot(df["Month"], df["Momentum_Score"], color=MANU_GOLD, lw=2)
    ax6.fill_between(df["Month"], df["Momentum_Score"],
                     df["Momentum_Score"].mean(),
                     where=(df["Momentum_Score"] >= df["Momentum_Score"].mean()),
                     alpha=0.3, color=MANU_RED)
    ax6.axhline(df["Momentum_Score"].mean(), color="grey", lw=1, ls="--",
                label=f"Mean={df['Momentum_Score'].mean():.1f}")
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    ax6.set_ylabel("Momentum Score")
    ax6.set_title("Monthly Momentum Score  (Win Rate × 0.4 + Goal Diff × 1.5 + Form × 0.5)")
    ax6.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig6)
    plt.close(fig6)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("9.  Statistical Analysis")

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
    st.subheader("Merged Monthly Dataset")
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

    st.subheader("Match-Level Raw Data")
    st.dataframe(mr, use_container_width=True, hide_index=True)

    st.subheader("Download")
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
st.caption(
    "Data: WRDS / CRSP (stock) · football-data.co.uk (matches) · "
    "Period: Aug 2019 – Dec 2024 · ACC102 Mini Assignment Track 4"
)
