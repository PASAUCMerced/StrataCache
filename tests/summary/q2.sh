#!/bin/bash

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What are the key differences between classical bits and quantum bits?"}]
  }'
