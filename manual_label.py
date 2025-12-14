import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# CONFIG (你只需要改这里)
# =========================
INPUT_CSV = "02_ai_scores_v2.csv"     # 你的 df 保存的 csv
TEXT_COL = "plain_text"
ID_COL = "orig_index"                # 强烈建议用 orig_index 作为唯一ID
SCORE_COL = "ai_score"
KEYWORD_COL = "keyword_score"        # 没有也行，会自动跳过
CLUSTER_COL = "cluster"              # 没有也行，会自动跳过

# 只标注 ai_score > 0 的候选集
FILTER_AI_SCORE_GT = 0.0

# 你要标注多少条（-1 表示全量候选）
LABEL_TARGET_N = 2000

# 抽样策略：stratified（分位数均匀抽样） or priority（优先低keyword/噪声/疑似假阳性）
SAMPLING_MODE = "priority"

# 输出
OUT_DIR = Path("results_human_label")
OUT_DIR.mkdir(exist_ok=True)
POOL_PATH = OUT_DIR / "label_pool.csv"
LABELS_PATH = OUT_DIR / "human_labels.csv"

# Topic taxonomy（你可以改）
TOPIC_LABELS = [
    "safety",
    "policy",
    "product_launch",
    "research",
    "ethics",
    "labour_market",
    "business",
    "auto_generation_content",
    "automation",
    "sports",
    "other",
]

# =========================
# Helpers
# =========================
def load_df():
    df = pd.read_csv(INPUT_CSV)
    if ID_COL not in df.columns:
        df[ID_COL] = df.index
    # 基本列检查
    assert TEXT_COL in df.columns, f"Missing {TEXT_COL}"
    assert SCORE_COL in df.columns, f"Missing {SCORE_COL}"
    return df

def build_label_pool(df: pd.DataFrame) -> pd.DataFrame:
    cand = df[df[SCORE_COL] > FILTER_AI_SCORE_GT].copy()

    # 只保留需要展示/标注的列（其余也可以加）
    keep_cols = [ID_COL, SCORE_COL, TEXT_COL]
    for c in ["title", "published_date", "publisher", "url", KEYWORD_COL, CLUSTER_COL]:
        if c in cand.columns:
            keep_cols.append(c)
    cand = cand[keep_cols].copy()

    # 去重（避免同一 orig_index 出现多次）
    cand = cand.drop_duplicates(subset=[ID_COL])

    # 抽样
    if LABEL_TARGET_N != -1 and len(cand) > LABEL_TARGET_N:
        rng = np.random.default_rng(42)

        if SAMPLING_MODE == "stratified":
            # 分位数均匀抽样，覆盖高分/低分
            cand["_q"] = pd.qcut(cand[SCORE_COL], q=10, duplicates="drop")
            per_bin = max(1, LABEL_TARGET_N // cand["_q"].nunique())
            parts = []
            for _, sub in cand.groupby("_q"):
                take = min(per_bin, len(sub))
                parts.append(sub.sample(n=take, random_state=42))
            pool = pd.concat(parts, ignore_index=True)
            if len(pool) > LABEL_TARGET_N:
                pool = pool.sample(n=LABEL_TARGET_N, random_state=42)
            pool = pool.drop(columns=["_q"], errors="ignore")
            return pool.reset_index(drop=True)

        # priority：优先挑“更可能是假阳性”的（你现在最需要清理）
        # 逻辑：keyword_score 越低越优先；有 cluster=-1 噪声也优先；ai_score 越接近阈值也优先
        pool = cand.copy()
        if KEYWORD_COL in pool.columns:
            pool["_kw"] = pool[KEYWORD_COL].fillna(0)
        else:
            pool["_kw"] = 0

        if CLUSTER_COL in pool.columns:
            pool["_noise"] = (pool[CLUSTER_COL] == -1).astype(int)
        else:
            pool["_noise"] = 0

        # “边界样本”更值得标（更能提高判别能力）
        pool["_margin"] = np.abs(pool[SCORE_COL] - 0.15)  # 你可以改中心点

        # 排序：先噪声、再低keyword、再靠近边界
        pool = pool.sort_values(
            by=["_noise", "_kw", "_margin"],
            ascending=[False, True, True]
        )

        pool = pool.head(LABEL_TARGET_N).drop(columns=["_kw", "_noise", "_margin"], errors="ignore")
        return pool.reset_index(drop=True)

    return cand.reset_index(drop=True)

def load_or_create_pool():
    if POOL_PATH.exists():
        return pd.read_csv(POOL_PATH)
    df = load_df()
    pool = build_label_pool(df)
    pool.to_csv(POOL_PATH, index=False, encoding="utf-8-sig")
    return pool

def load_labels():
    if LABELS_PATH.exists():
        lab = pd.read_csv(LABELS_PATH)
        if ID_COL not in lab.columns:
            return pd.DataFrame(columns=[ID_COL])
        return lab
    return pd.DataFrame(columns=[ID_COL])

def append_label(row_dict: dict):
    # 追加写入（断点续标）
    exists = LABELS_PATH.exists()
    out_df = pd.DataFrame([row_dict])
    out_df.to_csv(LABELS_PATH, mode="a", header=not exists, index=False, encoding="utf-8-sig")

# =========================
# Streamlit App
# =========================
def run_app():
    import streamlit as st

    st.set_page_config(page_title="Human Labeling: AI News", layout="wide")

    st.title("Human Labeling Tool (ai_score > 0 candidates)")
    st.caption(f"Pool: {POOL_PATH} | Labels: {LABELS_PATH}")

    pool = load_or_create_pool()
    labels = load_labels()

    labeled_ids = set(labels[ID_COL].tolist()) if len(labels) else set()

    # 未标注队列
    todo = pool[~pool[ID_COL].isin(labeled_ids)].copy()
    st.write(f"Total pool: {len(pool)} | Labeled: {len(labeled_ids)} | Remaining: {len(todo)}")

    if len(todo) == 0:
        st.success("All done. No remaining samples.")
        return

    # 选择当前样本
    if "cursor" not in st.session_state:
        st.session_state.cursor = 0

    cursor = st.session_state.cursor
    cursor = max(0, min(cursor, len(todo) - 1))
    st.session_state.cursor = cursor

    cur = todo.iloc[cursor]

    # 顶部控制条
    c1, c2, c3, c4, c5 = st.columns([1,1,2,2,2])
    with c1:
        if st.button("Prev"):
            st.session_state.cursor = max(0, cursor - 1)
            st.rerun()
    with c2:
        if st.button("Next"):
            st.session_state.cursor = min(len(todo) - 1, cursor + 1)
            st.rerun()
    with c3:
        jump = st.number_input("Jump to index", min_value=0, max_value=len(todo)-1, value=cursor, step=1)
        if st.button("Go"):
            st.session_state.cursor = int(jump)
            st.rerun()
    with c4:
        st.metric("Current", f"{cursor+1}/{len(todo)}")
    with c5:
        st.metric("orig_index", str(cur[ID_COL]))

    # 展示元信息
    meta_cols = []
    for c in ["title", "published_date", "publisher", "url", SCORE_COL, KEYWORD_COL, CLUSTER_COL]:
        if c in cur.index:
            meta_cols.append(c)

    if meta_cols:
        st.subheader("Metadata")
        st.json({c: (None if pd.isna(cur[c]) else cur[c]) for c in meta_cols})

    # 展示正文
    st.subheader("Article Text (plain_text)")
    st.text_area("",
                 value=str(cur[TEXT_COL])[:20000],
                 height=350)

    st.divider()
    st.subheader("Your Label")

    # 标注字段
    is_ai = st.radio("Is this article mainly about AI/ML/GPT/LLMs/automation?", ["yes", "no"], horizontal=True)
    topic = st.selectbox("Topic label", TOPIC_LABELS, index=TOPIC_LABELS.index("other") if "other" in TOPIC_LABELS else 0)

    has_barrier = st.checkbox("Mentions barriers/obstacles to AI adoption?")
    barrier_type = st.text_input("Barrier type (short, e.g. 'skills shortage', or leave blank)")
    notes = st.text_input("Notes (optional)")

    # 保存
    if st.button("Save label (and go next)", type="primary"):
        row = {
            ID_COL: cur[ID_COL],
            "human_is_ai": 1 if is_ai == "yes" else 0,
            "human_topic": topic,
            "human_has_barrier": 1 if has_barrier else 0,
            "human_barrier_type": barrier_type.strip() if barrier_type.strip() else "none",
            "notes": notes.strip(),
            "ts": int(time.time()),
        }
        append_label(row)
        st.success("Saved.")
        st.session_state.cursor = min(len(todo) - 1, cursor + 1)
        st.rerun()

    # 跳过（不写 label）
    if st.button("Skip (no save)"):
        st.session_state.cursor = min(len(todo) - 1, cursor + 1)
        st.rerun()

    st.divider()
    st.subheader("Quick export helpers")

    if st.button("Reload labels file"):
        st.rerun()

    st.write("Tip: you can stop anytime; labels are appended incrementally.")

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    run_app()
