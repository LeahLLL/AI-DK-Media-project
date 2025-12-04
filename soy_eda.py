import os
import pandas as pd
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import seaborn as sns
import plotly.express as px
import os
import matplotlib
matplotlib.use("Agg")
# Create required folders if not exist
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
# ===============================================================
# 0) GPU CHECK
# ===============================================================
if not torch.cuda.is_available():
    raise SystemError("❌ No GPU available. SBERT model requires GPU. Execution stopped!")

device = "cuda"

# Create folders
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ===============================================================
# 1) Load SBERT Model
# ===============================================================
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

# ===============================================================
# 2) Load Data
# ===============================================================
df = pd.read_csv("dk_news_2016_2024.csv")
df["plain_text"] = df["plain_text"].fillna("")
df.to_csv("results/00_raw_data.csv", index=False)

print("Loaded:", df.shape)

texts = df["plain_text"].astype(str).tolist()

# ===============================================================
# 3) Encode All News
# ===============================================================
print("Encoding articles via SBERT...")
emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)

np.save("results/01_embeddings.npy", emb)

# ===============================================================
# 4) AI Semantic Relevance Score
# ===============================================================
ai_query = """
AI, kunstig intelligens, maskinlæring, deep learning, LLM, GPT, robotter,
automatisering, dansk AI politik, AI etik, generativ AI.
"""

ai_vec = model.encode(ai_query, normalize_embeddings=True)
df["ai_score"] = util.cos_sim(emb, ai_vec).cpu().numpy()
df["is_ai"] = df["ai_score"] > 0.32
df.to_csv("results/02_ai_scores.csv", index=False)

print("AI-related articles:", df["is_ai"].sum())

# ===============================================================
# 5) AI Event Type Classification
# ===============================================================
event_queries = {
    "policy":     "The article discusses government policy or regulation of AI.",
    "investment": "The article discusses investment or funding related to AI.",
    "launch":     "The article reports release of a new AI product or service.",
    "ethics":     "The article discusses AI ethics, safety, or risks.",
    "research":   "The article reports AI research, breakthroughs, or innovation.",
    "industry":   "The article discusses AI impact on companies or industries.",
}

event_emb = model.encode(list(event_queries.values()), normalize_embeddings=True)
event_scores = util.cos_sim(emb, event_emb).cpu().numpy()

df["event_type"] = [list(event_queries.keys())[i] for i in event_scores.argmax(axis=1)]
df.to_csv("results/03_event_types.csv", index=False)

# ===============================================================
# 6) City Detection
# ===============================================================
city_names = ['København','Aarhus','Odense','Aalborg','Esbjerg','Randers','Kolding','Horsens','Vejle','Roskilde']
df["city"] = df["plain_text"].str.extract("(" + "|".join(city_names) + ")", flags=re.IGNORECASE)
df["city"] = df["city"].str.capitalize()
df["has_city"] = df["city"].notna()
df.to_csv("results/04_city_detection.csv", index=False)

# ===============================================================
# 7) Company Detection
# ===============================================================
company_list = [
    "Maersk", "Novo Nordisk", "Carlsberg", "Vestas",
    "Microsoft", "Google", "Amazon", "Meta", "OpenAI"
]

def detect_company(text: str):
    found = [c for c in company_list if re.search(rf"\b{c}\b", text, re.IGNORECASE)]
    return ", ".join(found)

df["company"] = df["plain_text"].apply(detect_company)
df["has_company"] = df["company"] != ""
df.to_csv("results/05_company_detection.csv", index=False)

# ===============================================================
# 8) YEAR
# ===============================================================
df["year"] = pd.to_datetime(df["published_date"]).dt.year
df.to_csv("results/06_with_year.csv", index=False)

# ===============================================================
# 9) Chart 1: AI NEWS TREND OVER TIME
# ===============================================================
trend = df.groupby("year")["is_ai"].sum()
trend.to_csv("results/07_ai_trend.csv")

plt.figure(figsize=(12,5))
trend.plot(marker="o")
plt.title("AI-related News Over Time (Denmark)")
plt.xlabel("Year")
plt.ylabel("Number of AI Articles")
plt.grid(True)
plt.savefig("figures/ai_trend.png")
plt.close()

# ===============================================================
# 10) Chart 2: AI Event Type Distribution
# ===============================================================
event_dist = df[df["is_ai"]]["event_type"].value_counts()
event_dist.to_csv("results/08_ai_event_distribution.csv")

plt.figure(figsize=(10,5))
event_dist.plot(kind="bar")
plt.title("AI Event Types")
plt.ylabel("Articles")
plt.savefig("figures/ai_event_types.png")
plt.close()

# ===============================================================
# 11) Chart 3: Companies Mentioned
# ===============================================================
comp_stats = (
    df[df["is_ai"] & df["has_company"]]
    .explode("company")
    .groupby("company")
    .size()
    .sort_values(ascending=False)
)

comp_stats.to_csv("results/09_ai_company_mentions.csv")

plt.figure(figsize=(10,5))
comp_stats.plot(kind="bar")
plt.title("Company Mentions in AI-related News")
plt.ylabel("Mentions")
plt.savefig("figures/ai_company_mentions.png")
plt.close()

# ===============================================================
# 12) Chart 4: AI-related News by City
# ===============================================================
city_ai = df[df["is_ai"] & df["has_city"]]["city"].value_counts()
city_ai.to_csv("results/10_ai_city_distribution.csv")

plt.figure(figsize=(10,5))
city_ai.plot(kind="bar")
plt.title("AI-related News by City")
plt.ylabel("Articles")
plt.savefig("figures/ai_city.png")
plt.close()

# ===============================================================
# 13) Chart 5: AI Topic Clustering
# ===============================================================
ai_emb = emb[df["is_ai"].values]
kmeans = KMeans(n_clusters=6, random_state=42).fit(ai_emb)

pca = PCA(n_components=2)
reduced = pca.fit_transform(ai_emb)

topic_df = pd.DataFrame({
    "pc1": reduced[:,0],
    "pc2": reduced[:,1],
    "cluster": kmeans.labels_
})
topic_df.to_csv("results/11_ai_cluster_pca.csv", index=False)

plt.figure(figsize=(10,7))
plt.scatter(reduced[:,0], reduced[:,1], c=kmeans.labels_, cmap="tab10", s=10)
plt.title("AI News Topic Clusters (SBERT PCA)")
plt.savefig("figures/ai_topic_clusters.png")
plt.close()

# ===============================================================
# 14) Chart 6: Word Cloud
# ===============================================================
all_ai_text = " ".join(df[df["is_ai"]]["plain_text"].tolist())

with open("results/12_wordcloud_text.txt", "w") as f:
    f.write(all_ai_text)

wc = WordCloud(width=1600, height=800, background_color="white").generate(all_ai_text)

plt.figure(figsize=(14,7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of AI-related News")
plt.savefig("figures/ai_wordcloud.png")
plt.close()

# ===============================================================
# 15) Chart 7: AI Score Distribution
# ===============================================================
plt.figure(figsize=(10,5))
sns.histplot(df["ai_score"], bins=40)
plt.title("AI Semantic Relevance Score Distribution")
plt.savefig("figures/ai_scores.png")
plt.close()

df[["ai_score"]].to_csv("results/13_ai_score_distribution.csv")

# ===============================================================
# 16) Chart 8: Company-Year Heatmap
# ===============================================================
heat = (
    df[df["is_ai"] & df["has_company"]]
    .groupby(["company", "year"])
    .size()
    .reset_index(name="count")
)
heat.to_csv("results/14_company_year_heatmap.csv", index=False)

fig = px.imshow(
    heat.pivot(index="company", columns="year", values="count"),
    title="AI Company Mentions Over Time"
)
fig.write_image("figures/ai_company_timeline.png")
# AI news ratio of Publisher Distribution of News Articles per Year in Denmark

# ===============================================================
# FINISHED
# ===============================================================
print("✔ ALL analysis completed.")
print("✔ All CSVs saved in ./results/")
print("✔ All figures saved in ./figures/")
