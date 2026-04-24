# Manchester United (MANU) — Stock Price & Match Performance Analysis

**ACC102 Mini Assignment – Track 4: Interactive Data Analysis Tool**

This repository contains a Streamlit web application that replicates and extends the data analysis pipeline for Manchester United (MANU). It explores the relationship between the club's on-pitch performance (Premier League match results) and its off-pitch financial performance (monthly stock returns).

## Features

The application is divided into six interactive tabs:

1. **Stock Overview**: Visualises the monthly closing price and returns, coloured by the dominant match result of that month. Includes cumulative return tracking and season-level financial summaries.
2. **Match Performance**: Breaks down match results (Win/Draw/Loss) overall and by season. Analyses home vs away performance, monthly points, and goal differences.
3. **Stock vs Match**: Explores the correlation between stock returns and various match metrics (Win Rate, Goal Difference, Momentum Score) using scatter plots with OLS regression lines and dual-axis timelines.
4. **EDA & Distributions**: Provides histograms and box plots of monthly returns, segmented by match results. Includes a correlation heatmap of all engineered features and a momentum score timeline.
5. **Statistical Tests**: Conducts formal statistical analysis, including One-Way ANOVA (returns across match results), Pearson correlations, Simple OLS Regression, and Independent t-Tests (Win vs Non-Win months).
6. **Raw Data**: Displays the merged monthly dataset and the raw match-level data, with options to download both as CSV files.

## Data Sources

- **Stock Data**: Monthly stock prices and returns sourced from WRDS / CRSP (`crsp.msf`).
- **Match Data**: Premier League match results sourced from [football-data.co.uk](https://www.football-data.co.uk).

*Note: The application uses pre-cleaned and aggregated data stored in the provided Excel files (`MANU_stock_WRDS_CRSP.xlsx` and `MANU_match_football_data.xlsx`).*

## Installation & Usage

### Prerequisites

Ensure you have Python 3.8 or higher installed.

### Setup

1. Clone this repository or download the source files.
2. Navigate to the project directory:
   ```bash
   cd manu_app
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the Streamlit app by running the following command in your terminal:

```bash
streamlit run app.py
```

The application will open automatically in your default web browser. If it does not, navigate to the local URL provided in the terminal output (usually `http://localhost:8501`).

## Project Structure

- `app.py`: The main Streamlit application script containing all data loading, feature engineering, and UI rendering logic.
- `requirements.txt`: List of Python dependencies required to run the app.
- `MANU_stock_WRDS_CRSP.xlsx`: Pre-processed monthly stock data.
- `MANU_match_football_data.xlsx`: Pre-processed match data (both monthly aggregated and raw match-level).
- `README.md`: This documentation file.
