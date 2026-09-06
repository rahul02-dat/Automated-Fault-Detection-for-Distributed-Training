#!/bin/bash

echo "Collecting Environment Info..."
echo "=============================="

echo "OS:"
uname -a

echo "Python version:"
python --version

echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"

echo "Git SHA:"
git rev-parse HEAD || echo "Not in a git repository"

echo "Environment collection completed."
