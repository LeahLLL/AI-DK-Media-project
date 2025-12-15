#!/usr/bin/env python
"""
vLLM sharded classifier for Danish news.

Adds HARD EVIDENCE requirement:
- If the model answers 'ja', it MUST also output:
  (a) AI terms explicitly found in the article (verbatim)
  (b) 1–2 short verbatim evidence quotes from the article (Danish)
Otherwise we auto-flip to 'nej' + score=0.

Outputs per article:

1) is_ai_llm                 : bool
2) ai_relevance_llm          : float (0–100)
3) ai_topic_llm              : str
4) ai_barrier_llm            : bool
5) ai_barrier_type_llm       : str
6) ai_barrier_summary_llm    : str
7) countries_llm             : str
8) cities_llm                : str
9) industries_llm            : str
10) ai_terms_llm             : str (comma-separated, verbatim from article or 'none')
11) ai_evidence_llm          : str (verbatim Danish quote(s) or 'none')

Input:
- CSV: 02_ai_scores_v2.csv
- TEXT_COL: plain_text
- requires keyword_score column (we filter keyword_score>0)

Sharding:
- ENV: START_ROW, NUM_SHARDS, SHARD_ID

Each shard writes:
    even_more_large_llm_<SHARD_ID>.csv
"""

import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ==========================
# Config
# ==========================

START_ROW = int(os.environ.get("START_ROW", "0"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "4"))
SHARD_ID   = int(os.environ.get("SHARD_ID", "0"))

# ---- Data paths ----
INPUT_PATH  = "selected_clusters.csv"
OUTPUT_PATH = f"selected_30B_{SHARD_ID}.csv"

TEXT_COL       = "plain_text"
ORIG_INDEX_COL = "orig_index"

# ---- Output columns ----
OUTPUT_COL_BOOL            = "is_ai_llm"
OUTPUT_COL_SCORE           = "ai_relevance_llm"
OUTPUT_COL_TOPIC           = "ai_topic_llm"
OUTPUT_COL_BARRIER_FLAG    = "ai_barrier_llm"
OUTPUT_COL_BARRIER_TYPE    = "ai_barrier_type_llm"
OUTPUT_COL_BARRIER_SUMMARY = "ai_barrier_summary_llm"
OUTPUT_COL_COUNTRIES       = "countries_llm"
OUTPUT_COL_CITIES          = "cities_llm"
OUTPUT_COL_INDUSTRIES      = "industries_llm"
OUTPUT_COL_AI_TERMS        = "ai_terms_llm"
OUTPUT_COL_AI_EVIDENCE     = "ai_evidence_llm"

# ---- Sharding ----
ENABLE_SHARDING = True

# ---- Row limit (for testing; -1 = use all after sharding) ----
ROW_LIMIT = -1

# ---- vLLM / model settings ----
MODEL_NAME           = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
TENSOR_PARALLEL_SIZE = 2
DTYPE                = "bfloat16"
MAX_MODEL_LEN        = 20000
GPU_MEM_UTIL         = 0.90
MAX_NUM_SEQS         = 32

# ---- Inference / batching ----
BATCH_SIZE           = 32
MAX_CHARS            = 20000
SAVE_EVERY_N_BATCHES = 10

# ---- Sampling ----
MAX_TOKENS  = 160    # more lines now, give enough budget
TEMPERATURE = 0.0
TOP_P       = 0.9

# ---- Retry for format errors ----
MAX_RETRY = 1  # retry only for malformed outputs


# ==========================
# Prompt & Parsing
# ==========================

def build_prompt(text: str) -> str:
    if text is None:
        text = ""
    text = text[:MAX_CHARS]

    return (
        "You are a strict AI news classifier. The news article below is written in Danish.\n"
        "You understand Danish, but you must always answer in English (except for 'ja'/'nej').\n\n"
        "Task:\n"
        "1) Decide if the article is mainly about artificial intelligence (AI), machine learning, "
        "GPT / large language models, or automation technology.\n"
        "2) Rate how strongly the article is related to AI on a scale from 0 to 100.\n"
        "3) Invent ONE short English topic label (1–3 words) describing the main AI-related theme.\n"
        "4) Decide whether the article discusses any barriers or obstacles related to AI "
        "(e.g., skills shortage, job loss concerns, regulation, ethical worries, technical challenges, costs).\n"
        "   If there is a barrier, invent ONE short English barrier label (1–3 words). "
        "   If there is no barrier, write 'none'.\n"
        "5) If there is a barrier, write ONE short sentence in English summarizing the barrier; else 'none'.\n"
        "6) Extract mentioned countries (0–5 items) from the article, in English, comma-separated; else 'none'.\n"
        "7) Extract mentioned cities/regions (0–5 items) from the article, in English, comma-separated; else 'none'.\n"
        "8) Extract mentioned industries/sectors (0–3 items) from the article, in English, comma-separated; "
        "   choose ONLY from: public sector, healthcare, finance, energy, manufacturing, transport, education, "
        "   retail, media, agriculture, IT/software, telecom, defense, legal, other. If none, write 'none'.\n"
        "9) Extract AI terms explicitly present in the text (0–6 items), VERBATIM as they appear in the article, "
        "   comma-separated; else 'none'. Examples of valid terms ONLY IF THEY APPEAR: "
        "   'kunstig intelligens', 'AI', 'maskinlæring', 'machine learning', 'GPT', 'LLM', 'chatbot', "
        "   'generativ AI', 'neural netværk', 'deep learning'.\n"
        "10) Provide 1–2 short VERBATIM evidence quote(s) from the article in Danish that justify your AI decision. "
        "   If you answer 'ja', this line MUST NOT be 'none'. If you answer 'nej', this line MUST be 'none'.\n\n"
        "CRITICAL RULES:\n"
        "- Output ONLY the required lines, no extra text.\n"
        "- Only extract items explicitly mentioned in the article. DO NOT infer.\n"
        "- If the main topic is AI/ML/GPT/automation → line 1 must be exactly: ja\n"
        "- Otherwise → line 1 must be exactly: nej\n"
        "- If you cannot find explicit AI terms AND explicit evidence quotes, you MUST answer 'nej'.\n\n"
        "Output format (ten lines):\n"
        "  Line 1: ja or nej\n"
        "  Line 2: a number between 0 and 100\n"
        "  Line 3: short English topic label (1–3 words)\n"
        "  Line 4: short English barrier label (or 'none')\n"
        "  Line 5: barrier summary in English (or 'none')\n"
        "  Line 6: countries (comma-separated English names or 'none')\n"
        "  Line 7: cities/regions (comma-separated English names or 'none')\n"
        "  Line 8: industries/sectors (comma-separated labels or 'none')\n"
        "  Line 9: AI terms VERBATIM from the article (comma-separated or 'none')\n"
        "  Line 10: VERBATIM Danish evidence quote(s) (or 'none')\n\n"
        "Article (in Danish):\n"
        f"{text}\n\n"
        "Answer:"
    )


def _norm_list_line(s: str) -> str:
    if not s:
        return "none"
    t = s.strip()
    if t.lower() in ("none", "no", "n/a", "na"):
        return "none"
    parts = [p.strip() for p in t.split(",") if p.strip()]
    if not parts:
        return "none"
    seen = set()
    out = []
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return ", ".join(out) if out else "none"


def _norm_evidence_line(s: str) -> str:
    if not s:
        return "none"
    t = s.strip()
    if t.lower() in ("none", "no", "n/a", "na"):
        return "none"
    # keep verbatim; just collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else "none"


def _is_valid_format(output_text: str) -> bool:
    if not output_text:
        return False
    lines = [ln.strip() for ln in output_text.splitlines() if ln.strip()]
    if len(lines) < 10:
        return False
    l1 = lines[0].lower()
    if not (l1.startswith("ja") or l1.startswith("nej")):
        return False
    # score line has a number
    if not re.search(r"\b\d+(\.\d+)?\b", lines[1]):
        return False
    return True


def parse_output(output_text: str):
    """
    Returns:
      (label_bool, score_float, topic_str,
       has_barrier_bool, barrier_type_str, barrier_summary_str,
       countries_str, cities_str, industries_str,
       ai_terms_str, ai_evidence_str)
    """
    if not output_text:
        return False, 0.0, "unknown", False, "none", "none", "none", "none", "none", "none", "none"

    raw = output_text.strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # defaults
    label = False
    score = 0.0

    # 1) label
    if len(lines) >= 1:
        l1 = lines[0].lower()
        if l1.startswith("ja"):
            label = True
        elif l1.startswith("nej"):
            label = False

    # 2) score
    score = 100.0 if label else 0.0
    if len(lines) >= 2:
        m = re.search(r"\b\d+(\.\d+)?\b", lines[1])
        if m:
            try:
                score = max(0.0, min(100.0, float(m.group(0))))
            except ValueError:
                score = 100.0 if label else 0.0

    # 3) topic
    topic = lines[2] if len(lines) >= 3 and lines[2] else "unknown"

    # 4) barrier type
    barrier_type = lines[3] if len(lines) >= 4 and lines[3] else "none"
    barrier_lower = barrier_type.lower()
    has_barrier = barrier_lower not in ("none", "no barrier", "no ai barrier", "no barriers")

    # 5) barrier summary
    barrier_summary = lines[4] if len(lines) >= 5 and lines[4] else "none"
    if not has_barrier:
        barrier_type = "none"
        barrier_summary = "none"

    # 6-8) lists
    countries  = _norm_list_line(lines[5]) if len(lines) >= 6 else "none"
    cities     = _norm_list_line(lines[6]) if len(lines) >= 7 else "none"
    industries = _norm_list_line(lines[7]) if len(lines) >= 8 else "none"

    # 9) ai terms, 10) evidence
    ai_terms    = _norm_list_line(lines[8]) if len(lines) >= 9 else "none"
    ai_evidence = _norm_evidence_line(lines[9]) if len(lines) >= 10 else "none"

    # --- HARD CONSISTENCY ENFORCEMENT ---
    # If label is ja, terms and evidence must exist; otherwise auto-flip to nej.
    if label:
        if ai_terms == "none" or ai_evidence == "none":
            label = False
            score = 0.0
            topic = "unknown"
            has_barrier = False
            barrier_type = "none"
            barrier_summary = "none"
            countries = "none"
            cities = "none"
            industries = "none"
            ai_terms = "none"
            ai_evidence = "none"
    else:
        # if nej, enforce none
        ai_terms = "none"
        ai_evidence = "none"

    return (label, score, topic, has_barrier, barrier_type, barrier_summary,
            countries, cities, industries, ai_terms, ai_evidence)


def generate_with_retry(llm: LLM, prompts, sampling_params, max_retry: int = 1):
    """
    Generate outputs with minimal retry for malformed formats.
    Returns list of generated texts aligned with prompts.
    """
    final = [""] * len(prompts)
    remaining = list(range(len(prompts)))
    cur_prompts = list(prompts)

    for attempt in range(max_retry + 1):
        outs = llm.generate([cur_prompts[i] for i in remaining], sampling_params)
        new_remaining = []

        for j, out in enumerate(outs):
            i = remaining[j]
            txt = out.outputs[0].text
            if _is_valid_format(txt):
                final[i] = txt
            else:
                new_remaining.append(i)

        if not new_remaining:
            break

        # strengthen instruction for failures
        for i in new_remaining:
            cur_prompts[i] = (
                "FORMAT ERROR: You MUST output exactly 10 non-empty lines, no extra text.\n"
                + cur_prompts[i]
            )
        remaining = new_remaining

    return final


# ==========================
# Main
# ==========================

def main():
    print(f"=== Shard {SHARD_ID}/{NUM_SHARDS} ===")
    print(f"Reading from: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    # # Keep your existing filter (keyword_score>0)
    # if "keyword_score" in df.columns:
    #     df = df[df["keyword_score"] > 0].copy()

    # Keep original index for sharding / later merging
    if ORIG_INDEX_COL not in df.columns:
        df[ORIG_INDEX_COL] = df.index
        df.reset_index(drop=True, inplace=True)

    # Apply START_ROW (optional) on orig_index
    if START_ROW > 0:
        df = df[df[ORIG_INDEX_COL] >= START_ROW].copy()

    # Sharding by orig_index % NUM_SHARDS == SHARD_ID
    if ENABLE_SHARDING:
        df = df[df[ORIG_INDEX_COL] % NUM_SHARDS == SHARD_ID].copy()

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
        OUTPUT_COL_COUNTRIES,
        OUTPUT_COL_CITIES,
        OUTPUT_COL_INDUSTRIES,
        OUTPUT_COL_AI_TERMS,
        OUTPUT_COL_AI_EVIDENCE,
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

    n_batches = (n_rows + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Start classification: batch_size={BATCH_SIZE}, total_batches={n_batches}")

    batch_counter = 0
    out_path = OUTPUT_PATH

    for start in tqdm(range(0, n_rows, BATCH_SIZE), desc=f"Shard {SHARD_ID}"):
        end = min(start + BATCH_SIZE, n_rows)
        batch = df.iloc[start:end]

        # Skip batch if already filled (resume)
        if (
            batch[OUTPUT_COL_BOOL].notna().all()
            and batch[OUTPUT_COL_SCORE].notna().all()
            and batch[OUTPUT_COL_TOPIC].notna().all()
            and batch[OUTPUT_COL_BARRIER_FLAG].notna().all()
            and batch[OUTPUT_COL_BARRIER_TYPE].notna().all()
            and batch[OUTPUT_COL_BARRIER_SUMMARY].notna().all()
            and batch[OUTPUT_COL_COUNTRIES].notna().all()
            and batch[OUTPUT_COL_CITIES].notna().all()
            and batch[OUTPUT_COL_INDUSTRIES].notna().all()
            and batch[OUTPUT_COL_AI_TERMS].notna().all()
            and batch[OUTPUT_COL_AI_EVIDENCE].notna().all()
        ):
            batch_counter += 1
            continue

        texts = batch[TEXT_COL].fillna("").tolist()
        prompts = [build_prompt(t) for t in texts]

        gen_texts = generate_with_retry(llm, prompts, sampling_params, max_retry=MAX_RETRY)

        bool_flags = []
        scores = []
        topics = []
        barrier_flags = []
        barrier_types = []
        barrier_summaries = []
        countries_list = []
        cities_list = []
        industries_list = []
        ai_terms_list = []
        ai_evidence_list = []

        for gen_text in gen_texts:
            (
                label,
                score,
                topic,
                has_barrier,
                barrier_type,
                barrier_summary,
                countries,
                cities,
                industries,
                ai_terms,
                ai_evidence,
            ) = parse_output(gen_text)

            bool_flags.append(label)
            scores.append(score)
            topics.append(topic)
            barrier_flags.append(has_barrier)
            barrier_types.append(barrier_type)
            barrier_summaries.append(barrier_summary)
            countries_list.append(countries)
            cities_list.append(cities)
            industries_list.append(industries)
            ai_terms_list.append(ai_terms)
            ai_evidence_list.append(ai_evidence)

        df.loc[batch.index, OUTPUT_COL_BOOL]            = bool_flags
        df.loc[batch.index, OUTPUT_COL_SCORE]           = scores
        df.loc[batch.index, OUTPUT_COL_TOPIC]           = topics
        df.loc[batch.index, OUTPUT_COL_BARRIER_FLAG]    = barrier_flags
        df.loc[batch.index, OUTPUT_COL_BARRIER_TYPE]    = barrier_types
        df.loc[batch.index, OUTPUT_COL_BARRIER_SUMMARY] = barrier_summaries
        df.loc[batch.index, OUTPUT_COL_COUNTRIES]       = countries_list
        df.loc[batch.index, OUTPUT_COL_CITIES]          = cities_list
        df.loc[batch.index, OUTPUT_COL_INDUSTRIES]      = industries_list
        df.loc[batch.index, OUTPUT_COL_AI_TERMS]        = ai_terms_list
        df.loc[batch.index, OUTPUT_COL_AI_EVIDENCE]     = ai_evidence_list

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

