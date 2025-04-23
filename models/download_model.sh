#!/bin/bash

# Install huggingface_hub
pip show huggingface_hub &> /dev/null || pip install huggingface_hub

# Download backbone model (Wav2Vec2-base)
echo "Download backbone: facebook/wav2vec2-base"
python3 - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/wav2vec2-base",
    local_dir="models/wav2vec2-base",
    local_dir_use_symlinks=False
)
EOF

# Download model and processor with huggingface_hub CLI
echo "Downloading wav2vec2-command-classifier..."
python3 - <<EOF
from huggingface_hub import snapshot_download

# Download everything on repo to local folder
snapshot_download(
    repo_id="Vorrapard/wav2vec2-command-classifier",
    local_dir="models/wav2vec2-command-classifier",
    local_dir_use_symlinks=False
)
EOF

echo "Models downloaded complete"
