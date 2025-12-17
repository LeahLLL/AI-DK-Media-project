DATA:https://huggingface.co/datasets/HamidBekam/dk_news_2016_2024
# DK News 2016-2024 Dataset
This dataset contains news articles from Denmark spanning the years 2016 to 2024. It is designed for use in natural language processing (NLP) tasks such as text classification, sentiment analysis, and topic modeling.

Github:https://github.com/LeahLLL/AI-DK-Media-project

data engineering:
folder: data engineering
-embedding_more.py : Script to add SBERT embeddings and some EDA to the dataset.
-tune_AI_score_and_more.ipynb : Notebook to tune AI detection scores and perform additional EDA.
-alz_submit.ipynb : Notebook for most paper result
-sentiment.py : Script for sentiment analysis on the dataset.
-Company_extraction.ipynb : Notebook to extract company names from the news articles.

LLM:
description: For AAU AI-Lab 8 Gpus vLLM classification
folder: LLM_classification
shard.py : actual running script for vLLM
shard.sh: bash script to run shard.py with different parameters
shard_start.sh: bash script to start multiple shard.sh with different parameters
aai_sbert.def : container settings for AAU AI-Lab

requirements.txt : required packages for the project

workflow: 
1. run embedding_more.py to add SBERT embeddings to the dataset.
2. use tune_AI_score_and_more.ipynb to tune AI detection scores and perform EDA.
3. use LLM classification scripts to classify the news articles using vLLM.
4. call alz_submit.ipynb, sentiment.py, Company_extraction.ipynb to generate results
