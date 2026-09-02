#!/bin/sh
# Put the pip-provided cuBLAS/cuDNN libraries where ctranslate2 can dlopen them, then run the worker.
set -e
LIBS=$(python -c "import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.pathsep.join([os.path.dirname(nvidia.cublas.lib.__file__), os.path.dirname(nvidia.cudnn.lib.__file__)]))")
export LD_LIBRARY_PATH="$LIBS:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi (CPU?)"
exec python /app/worker.py --loop
