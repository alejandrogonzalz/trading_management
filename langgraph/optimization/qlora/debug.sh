#!/bin/bash
# Interactive debug inside the container

cd ~/projects/trading_management/langgraph

echo "=== Entering container in debug mode ==="
echo "Inside the container you can run:"
echo "  - nvidia-smi"
echo "  - python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "  - wc -l backtest/data/labeled/dataset.jsonl"
echo "  - bash optimization/qlora/setup_docker_local.sh"
echo ""
echo "To exit: exit"
echo ""

docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 \
  -v "$(pwd):/workspace" -w /workspace \
  docker.io/unsloth/unsloth:latest \
  bash