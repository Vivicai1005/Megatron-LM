#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# This example will start serving the 345M model that is partitioned 8 way tensor parallel
DISTRIBUTED_ARGS="--nproc-per-node 4 \
                  --nnodes 1 \
                  --node-rank 0 \
                  --master-addr localhost \
                  --master-port 6000"

TOKENIZER_MODEL=/data/tigerbot/tigerbot_geely/test/cw/models/tigerbot-13b-chat-sofya-8k-v3-hf-1100-st-for-tgi/tokenizer.model
CHECKPOINT_DIR=/data/tigerbot/tigerbot_geely/test/cw/models/tigerbot-13b-chat-sofya-8k-v3-4tp

python -m torch.distributed.launch $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
       --tensor-model-parallel-size 4 \
       --pipeline-model-parallel-size 1 \
       --seq-length 4096 \
       --max-position-embeddings 4096 \
       --tokenizer-type Llama2Tokenizer \
       --tokenizer-model ${TOKENIZER_MODEL} \
       --load ${CHECKPOINT_DIR} \
       --exit-on-missing-checkpoint \
       --use-checkpoint-args \
       --no-load-optim \
       --no-load-rng \
       --fp16 \
       --untie-embeddings-and-output-weights \
       --use-rotary-position-embeddings \
       --normalization RMSNorm \
       --no-position-embedding \
       --no-masked-softmax-fusion \
       --micro-batch-size 1