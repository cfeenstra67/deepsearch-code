#!/usr/bin/env bash

out=benchmark-$(date +%s).json

./tictactoe.py \
    meta-llama/llama-4-scout \
    meta-llama/llama-4-scout::tools \
    meta-llama/llama-4-maverick \
    meta-llama/llama-4-maverick::tools \
    openai/gpt-4o \
    openai/gpt-4o::tools \
    mistralai/codestral-2501 \
    mistralai/codestral-2501::tools \
    openai/o3-mini-high \
    openai/o3-mini-high::tools \
    qwen/qwq-32b \
    qwen/qwq-32b::tools \
    qwen/qwen-turbo \
    qwen/qwen-turbo::tools \
    qwen/qwen2.5-vl-32b-instruct \
    qwen/qwen2.5-vl-32b-instruct::tools \
    google/gemini-2.5-pro-preview-03-25 \
    google/gemini-2.5-pro-preview-03-25::tools \
    mistralai/mistral-small-3.1-24b-instruct \
    mistralai/mistral-small-3.1-24b-instruct::tools \
    -n 20 \
    -a benchmarks.json \
    # deepseek/deepseek-r1 \
    # mistralai/codestral-2501 \
    # openai/o3-mini-high \
    # ai21/jamba-1.6-large \
    # qwen/qwq-32b \
    # anthropic/claude-3.7-sonnet:thinking \
    # qwen/qwen-turbo \
    # google/gemini-2.5-pro-preview-03-25 \
    # openrouter/quasar-alpha \
    # all-hands/openhands-lm-32b-v0.1 \
    # mistral/ministral-8b \
    # deepseek/deepseek-chat-v3-0324 \
    # openai/gpt-4o-mini \
    # google/gemini-2.0-flash-lite-001 \
    # google/gemini-2.0-flash-001 \
    # anthropic/claude-3.5-sonnet \
    # anthropic/claude-3.7-sonnet \
