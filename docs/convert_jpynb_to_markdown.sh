#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_notebook.ipynb> <output_markdown.md>"
  exit 1
fi

INPUT_NOTEBOOK="$1"
OUTPUT_MARKDOWN="$2"
OUTPUT_DIR=$(dirname "$OUTPUT_MARKDOWN")
OUTPUT_NAME=$(basename "$OUTPUT_MARKDOWN" .md)

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Convert notebook to markdown
jupyter nbconvert --to markdown "$INPUT_NOTEBOOK" --output "$OUTPUT_NAME" --output-dir "$OUTPUT_DIR"

echo "✅ Converted $INPUT_NOTEBOOK to $OUTPUT_MARKDOWN"
