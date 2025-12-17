# AI in Danish Media (2016–2024)

## Data
- **Hugging Face dataset:** https://huggingface.co/datasets/HamidBekam/dk_news_2016_2024  
  **DK News 2016–2024** is a Danish news corpus spanning 2016–2024, designed for NLP tasks such as text classification, sentiment analysis, and topic modeling.

## Repository
- **GitHub:** https://github.com/LeahLLL/AI-DK-Media-project

---

## Project Structure

### 1) Data Engineering (`data engineering/`)
Scripts and notebooks for embeddings, score tuning, EDA, and result generation.

- **`embedding_more.py`**  
  Add SBERT embeddings and run basic exploratory analysis (EDA).

- **`tune_AI_score_and_more.ipynb`**  
  Tune AI detection scores and perform additional EDA.

- **`alz_submit.ipynb`**  
  Main notebook used to generate most paper results.

- **`sentiment.py`**  
  Sentiment analysis pipeline for the dataset.

- **`Company_extraction.ipynb`**  
  Extract company/organization names from news articles.

---

### 2) LLM Classification (`LLM_classification/`)
Distributed vLLM classification on AAU AI-Lab (8 GPUs total; 2 GPUs per job), executed via sharding.

- **`shard.py`**  
  Main vLLM inference script (runs one shard/partition).

- **`shard.sh`**  
  Bash wrapper to run `shard.py` with different parameters.

- **`shard_start.sh`**  
  Launcher script to start multiple `shard.sh` jobs with different shard settings.

- **`aai_sbert.def`**  
  Container definition/settings for AAU AI-Lab execution.

---

## Dependencies
- **`requirements.txt`** — Required Python packages for the project.

---

## Workflow

1. **Generate embeddings**
   - Run `data engineering/embedding_more.py` to add SBERT embeddings to the dataset.

2. **Tune scores + EDA**
   - Use `data engineering/tune_AI_score_and_more.ipynb` to tune AI detection scores and run additional EDA.

3. **LLM classification (vLLM sharded jobs)**
   - Run scripts under `LLM_classification/` to classify news articles on AAU AI-Lab via sharding.

4. **Generate final results**
   - Use the following to produce paper-ready outputs:
     - `data engineering/alz_submit.ipynb`
     - `data engineering/sentiment.py`
     - `data engineering/Company_extraction.ipynb`

