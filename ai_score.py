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

keywords_query = ["AI", "kunstig intelligens", "maskinlæring", "dyb læring", "neural netværk", "automatisering",
                  "robotik", "dataanalyse", "algoritme", "intelligente systemer", "GPT", "OPENAI", "LLM", "chatbot",
                    "sprogmodel", "generativ AI", "AI-assistent", "AI-drevet", "computer vision", "naturlig sprogbehandling",
                    "AI-platform", "AI-teknologi", "AI-forskning", "AI-innovation", "AI-applikationer", "AI-løsninger",
                    "AI-udvikling", "AI-sikkerhed", "AI-etik", "AI-regulering", "AI-politik", "AI-strategi", "AI-investering",
                    "AI-startup", "AI-industrien", "AI-marked", "AI-trends", "AI-fremtid", "robotter", "automatiserede systemer",
                    "intelligente maskiner", "AI-integration", "AI-implementering", "AI-optimering", "AI-overvågning",]


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
keyword_scores = []
# key words search score
for index, row in df.iterrows():
    text = row['plain_text'].lower()
    # check if keywords exist in text
    keyword_score = sum(1 for kw in keywords_query if kw.lower() in text)
    keyword_scores.append(keyword_score)

keyword_scores = np.array(keyword_scores)
# normalize keyword scores
keyword_scores = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min() + 1e-10)
# assign to column
df['keyword_score'] = keyword_scores


df["ai_score"] = ai_final_score
df["is_ai"] = df["ai_score"] > 0
df.to_csv("results/02_ai_scores_v2.csv", index=False)

print("AI-related articles:", df["is_ai"].sum())

# ai score vs keyword score correlation
correlation = np.corrcoef(df["ai_score"], df["keyword_score"])[0, 1]
print("Correlation between AI score and Keyword score:", correlation)

# plot ai score vs keyword score distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(df["ai_score"], df["keyword_score"], alpha=0.5)
plt.title("AI Score vs Keyword Score Distribution")
plt.xlabel("AI Score")
plt.ylabel("Keyword Score")
plt.savefig("figures/ai_score_vs_keyword_score.png")
plt.show()

# random choose 200 plain text and fit it to ollama, to check if it is ai related or not
import random
sample_df_high_ai_score = df[df["ai_score"] > 0.1].sample(n=100, random_state=42)
sample_df_low_ai_score = df[df["ai_score"] <= -0.1].sample(n=100, random_state=42)
sample_df = pd.concat([sample_df_high_ai_score, sample_df_low_ai_score])
sample_df.to_csv("results/03_ai_sample_for_ollama.csv", index=False)
print("Sampled 200 articles for manual verification via Ollama: results/03_ai_sample_for_ollama.csv")

import ollama
model
def check_ai_via_ollama(text):
    prompt = f"is the following article about artificial intelligence (AI), machine learning, GPT, or automation? Answer with 'ja' for yes and 'nej' for no.\n\nArticle:\n{text}\n\nAnswer:"
    response = ollama.chat(model="llama2", prompt=prompt, max_tokens=10)
    answer = response['choices'][0]['message']['content'].strip().lower()
    return answer == 'ja'