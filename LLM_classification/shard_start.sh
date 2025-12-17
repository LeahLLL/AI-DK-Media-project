#!/bin/bash
# 一键提交多个 shard 任务，每个任务跑不同的 SHARD_ID

NUM_SHARDS=4

for SHARD_ID in $(seq 0 $((NUM_SHARDS-1))); do
  START_ROW=0   # 如果你没用到就统一 0，有需要也可以这里算不同起点

  echo "Submitting shard ${SHARD_ID}/${NUM_SHARDS} (START_ROW=${START_ROW})"

  sbatch \
    --export=ALL,START_ROW=${START_ROW},NUM_SHARDS=${NUM_SHARDS},SHARD_ID=${SHARD_ID} \
    shard.sh

done
