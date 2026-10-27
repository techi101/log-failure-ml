#!/usr/bin/env bash
# Download Loghub HDFS_v1 (once), run the tests, then the full ML pipeline.
# Usage: ./run.sh [data_dir]     (default: ./data/hdfs)
set -euo pipefail

DATA_DIR="${1:-data/hdfs}"
URL="https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"

if [ ! -f "$DATA_DIR/HDFS.log" ]; then
  mkdir -p "$DATA_DIR"
  echo "Downloading HDFS_v1 (~187 MB) ..."
  curl -L -o "$DATA_DIR/HDFS_v1.zip" "$URL"
  unzip -q -o "$DATA_DIR/HDFS_v1.zip" -d "$DATA_DIR"
fi

python -m pytest -q tests
python src/pipeline.py --data "$DATA_DIR"
echo "Results written to results/ (metrics.json + plots)"
