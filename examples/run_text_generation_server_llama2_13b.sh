#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# This example will start serving the 345M model that is partitioned 8 way tensor parallel
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TOKENIZER_MODEL=/data/tigerbot/tigerbot_geely/test/cw/models/tigerbot-13b-chat-sofya-8k-v3-hf-1100-st-for-tgi/tokenizer.model
CHECKPOINT_DIR=/data/tigerbot/tigerbot_geely/test/cw/models/tigerbot-13b-chat-sofya-8k-v3-1tp

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
       --tensor-model-parallel-size 1 \
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
       --attention-softmax-in-fp32 \
       --untie-embeddings-and-output-weights \
       --use-rotary-position-embeddings \
       --normalization RMSNorm \
       --no-position-embedding \
       --no-masked-softmax-fusion \
       --micro-batch-size 1  \
       --transformer-impl local 
