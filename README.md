DATA:https://huggingface.co/datasets/HamidBekam/dk_news_2016_2024
# DK News 2016-2024 Dataset
This dataset contains news articles from Denmark spanning the years 2016 to 2024. It is designed for use in natural language processing (NLP) tasks such as text classification, sentiment analysis, and topic modeling.

data engineering:
folder: data engineering
-embedding_more.py : Script to add SBERT embeddings and some EDA to the dataset.
-tune_AI_score_and_more.ipynb : Notebook to tune AI detection scores and perform additional EDA.
-alz_submit.ipynb : Notebook for most paper result

LLM:
description: For AAU AI-Lab 8 Gpus vLLM classification
folder: LLM_classification
shard.py : actual running script for vLLM
shard.sh: bash script to run shard.py with different parameters
shard_start.sh: bash script to start multiple shard.sh with different parameters
