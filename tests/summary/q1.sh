#!/bin/bash

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Quantum computing leverages quantum mechanical phenomena such as superposition and entanglement to perform computations. Unlike classical bits that represent either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously. This allows quantum computers to process vast amounts of information in parallel, potentially solving certain complex problems exponentially faster than classical computers. Key applications include cryptography, drug discovery, optimization problems, and materials science. However, quantum systems remain extremely sensitive to environmental noise, requiring sophisticated error correction and extremely low temperatures to maintain coherence."}]
  }'

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What are the key differences between classical bits and quantum bits?"}]
  }'

sleep 1

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What are the key differences between classical bits and quantum bits?"}]
  }'
