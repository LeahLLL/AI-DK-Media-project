import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
queries = [
    "Artiklen undersøger kunstig intelligens og machine learning.",
    "Teksten handler om AI-politik, regulering og etik.",
    "Dette dokument beskriver AI-teknologi, GPT og automatisering.",
    "Artiklen diskuterer brugen af robotter og generativ AI."
]

emb = np.load("results/01_embeddings.npy")
df = pd.read_csv("results/00_raw_data.csv")
ai_vec = model.encode(queries, normalize_embeddings=True)
query_vec = np.mean(ai_vec, axis=0)
negative_queries = [
    "Artiklen handler IKKE om kunstig intelligens, AI teknologi eller automatisering.",
    "Denne tekst drejer sig om almindelige nyheder, kultur, sport eller lokale begivenheder – ikke AI.",
    "Dokumentet indeholder ingen information om machine learning, GPT eller algoritmer.",
    "Teksten beskriver sociale, politiske eller menneskelige historier uden relation til AI eller IT teknologi."
]


neg_emb = model.encode(negative_queries, normalize_embeddings=True)
true_score =  util.cos_sim(emb, ai_vec) -  util.cos_sim(emb, neg_emb)
ai_final_score = true_score.mean(dim=1)


df["ai_score"] = ai_final_score
df["is_ai"] = df["ai_score"] > 0
df.to_csv("results/02_ai_scores_v2.csv", index=False)

print("AI-related articles:", df["is_ai"].sum())

# ai score distribution plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.hist(df['ai_score'], bins=50, color='blue', alpha=0.7)
plt.title('Distribution of AI Scores')
plt.xlabel('AI Score')
plt.ylabel('Number of Articles')
plt.savefig('figures/ai_score_distribution_1205.png')
plt.show()
