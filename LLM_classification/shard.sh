#!/bin/bash
#SBATCH --job-name=vllm_textgen
#SBATCH --output=logs/vllm_textgen_%A_%a.out
#SBATCH --error=logs/vllm_textgen_%A_%a.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=40G
#SBATCH --time=12:00:00

cd ~/semesterproject   # 建议加上，确保在正确目录
echo "Running shard with:"
echo "  START_ROW = ${START_ROW}"
echo "  NUM_SHARDS = ${NUM_SHARDS}"
echo "  SHARD_ID   = ${SHARD_ID}"
echo "  Host       = $(hostname)"

# 这些环境变量会自动传进容器里，shard_llm.py 用 os.environ.get(...) 就能读到
singularity exec --nv llm.sif python3 shard.py
