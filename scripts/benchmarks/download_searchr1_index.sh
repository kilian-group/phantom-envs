#!/usr/bin/env bash
# Download Search-R1's retrieval data from Hugging Face: the e5 FAISS index of Wikipedia 2018
# (PeterJinGo/wiki-18-e5-index, two parts joined into e5_Flat.index, ~64 GB) and its corpus
# (PeterJinGo/wiki-18-corpus, unzipped to corpus.jsonl, ~14 GB), named as the retriever expects.
#
# Usage:
#   bash scripts/benchmarks/download_searchr1_index.sh [data_dir]    (default: data/searchr1)
set -euo pipefail

DATA_DIR=${1:-data/searchr1}
mkdir -p "$DATA_DIR"

# Plain HTTP downloads: the xet protocol needs a writable $HF_HOME, which some shared setups lack
export HF_HUB_DISABLE_XET=1

echo "Downloading the FAISS index parts to $DATA_DIR ..."
uv run python - <<EOF
from huggingface_hub import hf_hub_download
for part in ["part_aa", "part_ab"]:
    hf_hub_download(repo_id="PeterJinGo/wiki-18-e5-index", filename=part, repo_type="dataset", local_dir="$DATA_DIR")
    print(f"Downloaded {part}")
EOF

echo "Joining the parts into e5_Flat.index ..."
cat "$DATA_DIR/part_aa" "$DATA_DIR/part_ab" > "$DATA_DIR/e5_Flat.index"
parts_size=$(( $(stat -c %s "$DATA_DIR/part_aa") + $(stat -c %s "$DATA_DIR/part_ab") ))
if [ "$(stat -c %s "$DATA_DIR/e5_Flat.index")" -ne "$parts_size" ]; then
    echo "Error: e5_Flat.index is smaller than its parts; kept the parts"
    exit 1
fi
rm "$DATA_DIR/part_aa" "$DATA_DIR/part_ab"

echo "Downloading the Wikipedia 2018 corpus ..."
uv run python - <<EOF
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="PeterJinGo/wiki-18-corpus", filename="wiki-18.jsonl.gz", repo_type="dataset", local_dir="$DATA_DIR")
EOF
gzip -d "$DATA_DIR/wiki-18.jsonl.gz"
mv "$DATA_DIR/wiki-18.jsonl" "$DATA_DIR/corpus.jsonl"

echo "Done:"
ls -lh "$DATA_DIR/e5_Flat.index" "$DATA_DIR/corpus.jsonl"
