import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------
# 0) Load all results from your results/ folder
# ----------------------------------------------------
# NOTE: Using a placeholder base_dir. Ensure this path is correct in your environment.
base_dir = r"C:\Users\qswwq\Documents\semesterproject\results"

df_raw         = pd.read_csv(os.path.join(base_dir, "00_raw_data.csv"))
df_ai_scores   = pd.read_csv(os.path.join(base_dir, "02_ai_scores.csv"))
df_event_types = pd.read_csv(os.path.join(base_dir, "03_event_types.csv"))
df_city        = pd.read_csv(os.path.join(base_dir, "04_city_detection.csv"))
df_company     = pd.read_csv(os.path.join(base_dir, "05_company_detection.csv"))
df_year        = pd.read_csv(os.path.join(base_dir, "06_with_year.csv"))

trend          = pd.read_csv(os.path.join(base_dir, "07_ai_trend.csv"), index_col=0)
event_dist     = pd.read_csv(os.path.join(base_dir, "08_ai_event_distribution.csv"), index_col=0)
comp_stats     = pd.read_csv(os.path.join(base_dir, "09_ai_company_mentions.csv"), index_col=0)
city_ai        = pd.read_csv(os.path.join(base_dir, "10_ai_city_distribution.csv"), index_col=0)
topic_pca      = pd.read_csv(os.path.join(base_dir, "11_ai_cluster_pca.csv"))
ai_scores      = pd.read_csv(os.path.join(base_dir, "13_ai_score_distribution.csv"))
heat           = pd.read_csv(os.path.join(base_dir, "14_company_year_heatmap.csv"))
MIN_YEAR = 2016
def clean_df(df):
    """If dataframe has publisher/title columns, clean it."""
    if 'publisher' in df.columns:
        df = df[df['publisher'] != 'sn.dk'].copy()
    if 'publisher' in df.columns and 'title' in df.columns:
        df = df.drop_duplicates(subset=['publisher', 'title'])
    return df

# Clean all dfs that contain news-level rows
df_raw         = clean_df(df_raw)
df_year        = clean_df(df_year)
df_ai_scores   = clean_df(df_ai_scores)
df_event_types = clean_df(df_event_types)
df_city        = clean_df(df_city)
df_company     = clean_df(df_company)
# Filter the primary yearly dataframe
df_year = df_year[df_year['year'] >= MIN_YEAR].copy()

# Filter the aggregated trend data (index is year)
trend = trend[trend.index >= MIN_YEAR].copy()

# Filter the aggregated heatmap data (columns are year)
# Need to convert column names to integer first if they are strings
# Note: heat columns should typically be converted to str for the pivot/imshow to work,
# but since the column names are years, we convert them to int temporarily for filtering.
try:
    heat.columns = heat.columns.astype(int)
    columns_to_keep = [col for col in heat.columns if col >= MIN_YEAR]
    heat = heat[columns_to_keep].copy()
    # Convert back to string for pivot/imshow if needed, though px.imshow often handles int columns fine
    heat.columns = heat.columns.astype(str)
except ValueError:
    # Handles cases where column names might not all be integers (e.g., 'company')
    # Assuming 'company' is the index and columns are years
    pass
# Merge everything into one df for convenience
df = df_year.copy()

# ----------------------------------------------------------------------
# FIX: Convert index-based data (loaded with index_col=0) to clean
# DataFrames with named columns for Plotly Express compatibility.
# ----------------------------------------------------------------------

# 1) AI Trend Over Time
trend_df = trend.reset_index()
# Assuming the index was 'year' and the value column was 'is_ai' (or similar)
trend_df.columns = ["Year", "Number of AI Articles"]
fig1 = px.line(
    trend_df,
    x="Year",
    y="Number of AI Articles",
    markers=True,
    title="AI-related News Over Time (Denmark)",
)
fig1.update_layout(template="plotly_white")
fig1.show()

# 2) AI Event Distribution
event_df = event_dist.reset_index()
# The resulting DataFrame will have two columns: the old index and the count data
event_df.columns = ["Event Type", "Articles"]
fig2 = px.bar(
    event_df,
    x="Event Type",
    y="Articles",
    title="AI Event Types",
)
fig2.update_layout(template="plotly_white")
fig2.show()

# 3) Company Mentions
comp_df = comp_stats.reset_index()
comp_df.columns = ["Company", "Mentions"]
fig3 = px.bar(
    comp_df,
    x="Company",
    y="Mentions",
    title="Company Mentions in AI-related News",
)
fig3.update_layout(template="plotly_white")
fig3.show()

# 4) AI-related News by City
city_df = city_ai.reset_index()
city_df.columns = ["City", "Articles"]
fig4 = px.bar(
    city_df,
    x="City",
    y="Articles",
    title="AI-related News by City",
)
fig4.update_layout(template="plotly_white")
fig4.show()

# ----------------------------------------------------
# 5) Chart: PCA Cluster Scatter (Original code was fine)
# ----------------------------------------------------
fig5 = px.scatter(
    topic_pca,
    x="pc1",
    y="pc2",
    color="cluster",
    color_continuous_scale="Turbo",
    title="AI News Topic Clusters (SBERT PCA)",
    labels={"pc1": "PC1", "pc2": "PC2"}
)
fig5.update_layout(template="plotly_white")
fig5.show()

# ----------------------------------------------------
# 6) Chart: AI Score Distribution (Original code was fine)
# ----------------------------------------------------
fig6 = px.histogram(
    ai_scores,
    x="ai_score",
    nbins=40,
    title="AI Semantic Relevance Score Distribution",
    labels={"ai_score": "AI Score"}
)
fig6.update_layout(template="plotly_white")
fig6.show()

# ----------------------------------------------------
# 7) Company-Year Heatmap (Original code was fine)
# ----------------------------------------------------
pivot = heat.pivot(index="company", columns="year", values="count")

fig7 = px.imshow(
    pivot,
    title="AI Company Mentions Over Time",
    labels=dict(x="Year", y="Company", color="Count"),
    aspect="auto"
)
fig7.update_layout(template="plotly_white")
fig7.show()
# ----------------------------------------------------
# 8) Chart: AI Prominence (AI Articles / Total Articles)
# ----------------------------------------------------

# 1. Calculate total articles per year from the main yearly dataset (df_year)
all_topic_year_series = df_year['year'].value_counts().sort_index().rename('Total Articles')
all_topic_year_df = all_topic_year_series.reset_index()
all_topic_year_df.columns = ["Year", "Total Articles"]

# 2. Prepare AI Articles per Year (from the 'trend' data)
ai_topic_year_df = trend.reset_index()
# Assuming the column name in trend is 'is_ai' (or similar, it was indexed 0 when loaded)
# and contains the count of AI articles.
ai_topic_year_df.columns = ["Year", "AI Articles"]

# 3. Merge the two datasets on 'Year'
prominence_df = pd.merge(all_topic_year_df, ai_topic_year_df, on="Year", how="inner")

# 4. Calculate the ratio
# Avoid division by zero by setting the ratio to 0 if Total Articles is 0
prominence_df['AI Prominence Ratio'] = prominence_df.apply(
    lambda row: row['AI Articles'] / row['Total Articles'] if row['Total Articles'] > 0 else 0,
    axis=1
)

# 5. Create the Plot
fig8 = px.line(
    prominence_df,
    x="Year",
    y="AI Prominence Ratio",
    markers=True,
    title="Prominence of AI-related News Over Time (AI Articles / Total Articles)",
    labels={"AI Prominence Ratio": "Ratio (AI Articles / Total Articles)"}
)
fig8.update_layout(template="plotly_white", yaxis_tickformat=".2%") # Format y-axis as percentage
fig8.show()

import plotly.graph_objects as go

year_total = df_year.groupby("year")["title"].count().rename("Total Articles")
year_ai = df_year[df_year["is_ai"] == True].groupby("year")["title"].count().rename("AI Articles")

df_ratio = pd.concat([year_total, year_ai], axis=1).fillna(0)
df_ratio["AI Ratio"] = df_ratio["AI Articles"] / df_ratio["Total Articles"]

df_ratio = df_ratio.reset_index()

# ----------------------------------------------------
# 3) Plot: Total Articles + AI Ratio
# ----------------------------------------------------
fig = go.Figure()

# A) Bar — total articles
fig.add_trace(
    go.Bar(
        x=df_ratio["year"],
        y=df_ratio["Total Articles"],
        name="Total Articles",
        marker=dict(opacity=0.7)
    )
)

# B) Line — AI ratio
fig.add_trace(
    go.Scatter(
        x=df_ratio["year"],
        y=df_ratio["AI Ratio"],
        mode="lines+markers",
        name="AI Ratio",
        yaxis="y2"
    )
)

fig.update_layout(
    title=f"AI News Ratio & Distribution of Articles per Year ",
    xaxis_title="Year",
    yaxis_title="Total Articles",
    yaxis2=dict(
        title="AI Ratio",
        overlaying="y",
        side="right",
        tickformat=".1%"
    ),
    template="plotly_white",
    bargap=0.15
)

fig.show()


# compute per-publisher per-year stats
df_pub = df_year.copy()

# remove publisher only containing less than 50 articles in total per year
publisher_counts = df_pub['publisher'].value_counts()
valid_publishers = publisher_counts[publisher_counts >= 50].index
df_pub = df_pub[df_pub['publisher'].isin(valid_publishers)]

grouped = df_pub.groupby(["publisher", "year"]) \
    .agg(
        Total_Articles=("title", "count"),
        AI_Articles=("is_ai", "sum")
    ).reset_index()
import numpy as np
grouped["AI_Ratio"] = grouped["AI_Articles"] / grouped["Total_Articles"]
# average AI ratio per publisher output
grouped = grouped.sort_values(by=["year", "publisher"])


fig = px.bar(
    grouped,
    x="year",
    y="AI_Ratio",
    color="publisher",
    title="Publisher Distribution of News Articles per Year in Denmark",
)

# Publisher ratio Distribution of News Articles per Year in Denmark
df_pub = df_year.copy()

# remove publisher only containing less than 50 articles in total per year
publisher_counts = df_pub['publisher'].value_counts()
valid_publishers = publisher_counts[publisher_counts >= 50].index
df_pub = df_pub[df_pub['publisher'].isin(valid_publishers)]

grouped = df_pub.groupby(["publisher", "year"]) \
    .agg(
        Total_Articles=("title", "count"),
        AI_Articles=("is_ai", "sum")
    ).reset_index()
import numpy as np
grouped["AI_Ratio"] = grouped["AI_Articles"] / grouped["Total_Articles"]
# average AI ratio per publisher output
grouped = grouped.sort_values(by=["year", "publisher"])


fig = px.bar(
    grouped,
    x="year",
    y="AI_Ratio",
    color="publisher",
    title="Publisher Distribution of News Articles per Year in Denmark",
)

fig.update_layout(
    template="plotly_white",
    barmode="group",        # ← 把 stack 改成 group，不要堆叠
    xaxis_title="Year",
    yaxis_title="AI Ratio",
    legend_title="Publisher",
    hovermode="x unified",
    yaxis_tickformat=".1%",  # ← 百分比格式
    width=2000,
    height=800,
)

fig.show()

