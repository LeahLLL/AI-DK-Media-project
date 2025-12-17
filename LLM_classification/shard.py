#!/usr/bin/env python
"""
vLLM + Gemma3 sharded classifier for Danish news.

For each Danish news article, the model outputs:

1) is_ai_llm              : bool  (True if article mainly about AI/ML/GPT/automation)
2) ai_relevance_llm       : float (0–100 relevance score)
3) ai_topic_llm           : str   (free English topic label, 1–3 words)
4) ai_barrier_llm         : bool  (True if article discusses a barrier/obstacle of AI)
5) ai_barrier_type_llm    : str   (free English barrier label, 1–3 words or 'none')
6) ai_barrier_summary_llm : str   (one-sentence English summary of the barrier, or 'none')

Input:
- CSV file: dk_news_2016_2024.csv
- Text column: 'plain_text' (Danish text)

Sharding:
- ENV: START_ROW (optional, default 0)
- ENV: NUM_SHARDS (default 4)
- ENV: SHARD_ID   (default 0)
- If ENABLE_SHARDING = True, we only keep rows where orig_index % NUM_SHARDS == SHARD_ID

Usage:
- Set up a sbatch like:
    sbatch --export=ALL,START_ROW=0,NUM_SHARDS=4,SHARD_ID=0 run_shard.sbatch
- Each shard writes its own CSV:
    dk_news_2016_2024_ai_shard_<SHARD_ID>.csv
"""

import os
import re
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ==========================
# Config
# ==========================



TEXT_COL       = "plain_text"
ORIG_INDEX_COL = "orig_index"   # keep original index for sharding & later merge

# ---- Output columns ----
OUTPUT_COL_BOOL            = "is_ai_llm"
OUTPUT_COL_SCORE           = "ai_relevance_llm"
OUTPUT_COL_TOPIC           = "ai_topic_llm"
OUTPUT_COL_BARRIER_FLAG    = "ai_barrier_llm"
OUTPUT_COL_BARRIER_TYPE    = "ai_barrier_type_llm"
OUTPUT_COL_BARRIER_SUMMARY = "ai_barrier_summary_llm"

# ---- Sharding ----
ENABLE_SHARDING = True
START_ROW = int(os.environ.get("START_ROW", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "4"))
SHARD_ID   = int(os.environ.get("SHARD_ID", "0"))

# ---- Data paths ----
INPUT_PATH  = "dk_news_2016_2024.csv"
OUTPUT_PATH = f"dk_news_2016_2024_ai_shard_{SHARD_ID}_v2.csv"
# ---- Row limit (for testing; -1 = use all after sharding) ----
ROW_LIMIT = -1   # e.g. set to 5000 for quick tests; keep -1 for full run

# ---- vLLM / model settings ----
MODEL_NAME           = "Qwen/Qwen3-VL-8B-Instruct"
TENSOR_PARALLEL_SIZE = 2      # you said each job uses 2 GPUs
DTYPE                = "bfloat16"
MAX_MODEL_LEN        = 20000   # enough for ~6000 Danish chars + prompt
GPU_MEM_UTIL         = 0.90
MAX_NUM_SEQS         = 32     # max parallel sequences inside vLLM engine

# ---- Inference / batching ----
BATCH_SIZE           = 32     # you can tune this up/down depending on VRAM
MAX_CHARS            = 20000   # truncate article text (Danish) to this many chars
SAVE_EVERY_N_BATCHES = 10    # checkpoint frequency (batches)

# ---- Sampling ----
MAX_TOKENS  = 48    # output: 5 lines, including a short sentence → give some slack
TEMPERATURE = 0.0
TOP_P       = 0.9


# ==========================
# Prompt & Parsing
# ==========================

def build_prompt(text: str) -> str:
    """
    Build the LLM prompt.

    IMPORTANT:
    - The article is in DANSK (Danish).
    - The model should READ Danish, but OUTPUT labels & summary in ENGLISH
      (except for 'ja' / 'nej').
    """
    if text is None:
        text = ""
    text = text[:MAX_CHARS]

    return (
        "You are a strict AI news classifier. The news article below is written in Danish.\n"
        "You understand Danish, but you must always answer in English (except for 'ja'/'nej').\n\n"
        "Task: For the following Danish news article, you must:\n"
        "1) Decide if it is mainly about artificial intelligence (AI), "
        "machine learning, GPT / large language models, or automation technology.\n"
        "2) Rate how strongly the article is related to AI on a scale from 0 to 100.\n"
        "3) Invent ONE short English topic label (1–3 words) describing the main AI-related theme.\n"
        "   Examples: 'AI safety', 'AI policy', 'product launch', 'AI research', "
        "'ethics', 'labour market', 'business strategy', 'auto content', 'automation', 'sports', etc.\n"
        "4) Decide whether the article discusses any barriers or obstacles related to AI "
        "(for example: skills shortage, job loss concerns, regulation, ethical worries, "
        "technical challenges, costs, lack of trust, etc.). "
        "If there is a barrier, invent ONE short English barrier label (1–3 words). "
        "If there is no barrier, write 'none'.\n\n"
        "Non_ai example:Det holdt hårdt, men Danmark vandt alligevel 1-0 i fredagens EM-kvalifikationskamp mod Nordirland. I tillægstiden havde Nordirland bolden i nettet til en udligning, men målet blev kaldt tilbage for offside efter en længere VAR-gennemgang. Pierre-Emile Højbjerg gav kort efter slutfløjt en kommentar til TV 2, og han mente, der var klasseforskel på de to hold, selvom Danmark var heldige i sidste ende. - Der er ingen nemme fodboldkampe. En 1-0-sejr er selvfølgelig dejligt, men jeg er skuffet over de sidste 10 minutter, hvor vi ikke fik fat som hold. Vi var faktisk rigtig heldige. Artiklen fortsætter efter billedet - Set over hele kampen synes jeg, vi var klart et bedre fodboldhold. Jeg synes også, vi viste modenhold. - Det er altid nemt at være det lille hold, for de skal bare bruge én chance. De spiller for en standardsituation eller en løs bold, udtalte Højbjerg til TV 2. Med sejren er Danmark på førstepladsen i Gruppe H. Finland, Kasakhstan og Slovenien er ligesom Danmark på seks point efter tre kampe.\n\n"
        "Important rules:\n"
        "- If the main topic is AI/ML/GPT/automation → answer exactly: ja\n"
        "- Otherwise → answer exactly: nej\n"
        "- WARNING: If the content is just automatically generated BY AI but is NOT ABOUT "
        "AI/ML/GPT/automation itself, then the answer is 'nej'.\n"
        "- The AI relevance score must be a single number between 0 and 100 "
        "(0 = not related at all, 100 = strongly focused on AI).\n"
        "- The topic label must be a short English phrase (1–3 words), no extra explanation.\n"
        "- The barrier label must be a short English phrase (1–3 words), "
        "or 'none' if there is no AI-related barrier.\n"
        "- If there is a barrier, write a short one-sentence summary in English describing "
        "the barrier (what is difficult or blocking AI). If there is no barrier, write 'none'.\n\n"
        "Output format (five lines):\n"
        "  Line 1: ja or nej\n"
        "  Line 2: a number between 0 and 100\n"
        "  Line 3: free topic label (short English phrase)\n"
        "  Line 4: free barrier label (short English phrase or 'none')\n"
        "  Line 5: barrier summary in English (or 'none' if there is no barrier)\n\n"
        "Article (in Danish):\n"
        f"{text}\n\n"
        "Answer:"
    )


def parse_label_score_topic_barrier(output_text: str):
    """
    Parse 5-line output:
      Line 1: ja/nej
      Line 2: 0–100 (AI relevance)
      Line 3: free topic label (string)
      Line 4: free barrier label (string, 'none' means no barrier)
      Line 5: barrier summary (string)

    Returns:
      (label_bool, score_float, topic_str, has_barrier_bool, barrier_type_str, barrier_summary_str)
    """
    if not output_text:
        return False, 0.0, "unknown", False, "none", "none"

    raw = output_text.strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    lower_lines = [ln.lower() for ln in lines]

    # 1) label: ja / nej
    label = False
    if lower_lines:
        if lower_lines[0].startswith("ja"):
            label = True
        elif lower_lines[0].startswith("nej"):
            label = False

    # 2) score: first number in text, clamped to [0,100]
    score = 0.0
    m = re.search(r"\b\d+(\.\d+)?\b", raw.lower())
    if m:
        try:
            val = float(m.group(0))
            if val < 0:
                val = 0.0
            elif val > 100:
                val = 100.0
            score = val
        except ValueError:
            score = 100.0 if label else 0.0
    else:
        score = 100.0 if label else 0.0

    # 3) topic label: line 3
    topic = "unknown"
    if len(lines) >= 3:
        topic = lines[2].strip()
        if not topic:
            topic = "unknown"

    # 4) barrier label: line 4; 'none' / 'no barrier' → no barrier
    barrier_type = "none"
    has_barrier = False
    if len(lines) >= 4:
        barrier_type = lines[3].strip()
        if not barrier_type:
            barrier_type = "none"

    barrier_lower = barrier_type.lower()
    if barrier_lower in ("none", "no barrier", "no ai barrier", "no barriers"):
        has_barrier = False
        barrier_type = "none"
    else:
        has_barrier = True

    # 5) barrier summary: line 5; if no barrier → 'none'
    barrier_summary = "none"
    if len(lines) >= 5:
        barrier_summary = lines[4].strip() or "none"

    if not has_barrier:
        barrier_summary = "none"

    return label, score, topic, has_barrier, barrier_type, barrier_summary


# ==========================
# Main
# ==========================

def main():
    print(f"=== Shard {SHARD_ID}/{NUM_SHARDS} ===")
    print(f"Reading from: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    # Keep original index for sharding / later merging
    if ORIG_INDEX_COL not in df.columns:
        df[ORIG_INDEX_COL] = df.index

    # Apply START_ROW (optional)
    if START_ROW > 0:
        df = df[df.index >= START_ROW].copy()

    # Sharding by orig_index % NUM_SHARDS == SHARD_ID
    if ENABLE_SHARDING:
        df = df[df.index % NUM_SHARDS == SHARD_ID].copy()

    # Apply ROW_LIMIT *after* sharding, if > 0
    if ROW_LIMIT > 0:
        df = df.iloc[:ROW_LIMIT].copy()

    df = df.reset_index(drop=True)
    n_rows = len(df)
    print(f"Rows in this shard: {n_rows}")

    if n_rows == 0:
        print("No rows in this shard, exiting.")
        return

    # Ensure output columns exist
    for col in [
        OUTPUT_COL_BOOL,
        OUTPUT_COL_SCORE,
        OUTPUT_COL_TOPIC,
        OUTPUT_COL_BARRIER_FLAG,
        OUTPUT_COL_BARRIER_TYPE,
        OUTPUT_COL_BARRIER_SUMMARY,
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    # Init LLM
    print("Initializing LLM engine...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_num_seqs=MAX_NUM_SEQS,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    # Batch inference
    n_batches = (n_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(
        f"Start classification: "
        f"batch_size={BATCH_SIZE}, total_batches={n_batches}"
    )

    batch_counter = 0
    out_path = OUTPUT_PATH.format(shard=SHARD_ID)

    for start in tqdm(range(0, n_rows, BATCH_SIZE), desc=f"Shard {SHARD_ID}"):
        end = min(start + BATCH_SIZE, n_rows)
        batch = df.iloc[start:end]

        # Skip batch if all outputs are already filled (for resume)
        if (
            batch[OUTPUT_COL_BOOL].notna().all()
            and batch[OUTPUT_COL_SCORE].notna().all()
            and batch[OUTPUT_COL_TOPIC].notna().all()
            and batch[OUTPUT_COL_BARRIER_FLAG].notna().all()
            and batch[OUTPUT_COL_BARRIER_TYPE].notna().all()
            and batch[OUTPUT_COL_BARRIER_SUMMARY].notna().all()
        ):
            batch_counter += 1
            continue

        texts = batch[TEXT_COL].fillna("").tolist()
        prompts = [build_prompt(t) for t in texts]

        outputs = llm.generate(prompts, sampling_params)

        bool_flags = []
        scores = []
        topics = []
        barrier_flags = []
        barrier_types = []
        barrier_summaries = []

        for out in outputs:
            gen_text = out.outputs[0].text
            (
                label,
                score,
                topic,
                has_barrier,
                barrier_type,
                barrier_summary,
            ) = parse_label_score_topic_barrier(gen_text)

            bool_flags.append(label)
            scores.append(score)
            topics.append(topic)
            barrier_flags.append(has_barrier)
            barrier_types.append(barrier_type)
            barrier_summaries.append(barrier_summary)

        df.loc[batch.index, OUTPUT_COL_BOOL]            = bool_flags
        df.loc[batch.index, OUTPUT_COL_SCORE]           = scores
        df.loc[batch.index, OUTPUT_COL_TOPIC]           = topics
        df.loc[batch.index, OUTPUT_COL_BARRIER_FLAG]    = barrier_flags
        df.loc[batch.index, OUTPUT_COL_BARRIER_TYPE]    = barrier_types
        df.loc[batch.index, OUTPUT_COL_BARRIER_SUMMARY] = barrier_summaries

        batch_counter += 1

        if SAVE_EVERY_N_BATCHES > 0 and batch_counter % SAVE_EVERY_N_BATCHES == 0:
            tmp_out = out_path + ".tmp"
            print(f"\nCheckpoint: saving shard {SHARD_ID} to {tmp_out}")
            df.to_csv(tmp_out, index=False)

    print(f"\nSaving final shard {SHARD_ID} to: {out_path}")
    df.to_csv(out_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
