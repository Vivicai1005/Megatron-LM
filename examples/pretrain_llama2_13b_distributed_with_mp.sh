#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/data/tigerbot/tigerbot_geely/test/cw/models/tigerbot-13b-chat-sofya-8k-v3-4tp
DATA_PATH=/data/tigerbot/tigerbot_geely/test/cw/data/snomed_train_ner/snomed_train_ner_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --finetune
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 13824 \
    --num-attention-heads 40 \
    --micro-batch-size 1 \
    --global-batch-size 1  \
    --lr 6.0e-5 \
    --lr-decay-iters 10 \
    --lr-warmup-iters 5 \
    --min-lr 6.0e-6 \
    --override-opt_param-scheduler \
    --lr-decay-style cosine \
    --train-iters 1000 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --no-gradient-accumulation-fusion \
    --transformer-impl local
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 98,2,0
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model /data/tigerbot/tigerbot_geely/test/cw/models/tigerbot-13b-chat-sofya-8k-v3-hf-1100-st-for-tgi/tokenizer.model \
    --exit-on-missing-checkpoint \
    --use-checkpoint-args \
    --no-load-optim \
    --no-load-rng \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --no-position-embedding \
    --swiglu \
    --disable-bias-linear \
    --no-masked-softmax-fusion \
    --bf16 \
    --recompute-activations \
    --recompute-granularity selective \
    --attention-softmax-in-fp32 \
    --use-flash-attn \
    --eod-mask-loss

