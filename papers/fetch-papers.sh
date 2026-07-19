#!/usr/bin/env bash
# Fetch the reading copies (PDFs) for every paper in this repo from arXiv.
# PDFs and arXiv source trees are deliberately NOT in git (see .gitignore) —
# run this once after cloning. Safe to re-run: existing files are skipped.
#
# Usage:
#   ./papers/fetch-papers.sh            # fetch all missing PDFs
#   ./papers/fetch-papers.sh 06-TFN     # fetch one paper's PDF
#
# To also grab a paper's LaTeX source (for reMarkable reformatting etc.):
#   curl -L "https://arxiv.org/e-print/<id>" | tar xz -C papers/<folder>/arXiv-<id>/
#
# The folder <-> arXiv ID mapping mirrors references/papers.md — update both
# together when adding a paper.

set -euo pipefail
cd "$(dirname "$0")"

# folder:arxiv_id:pdf_name
PAPERS="
01-EGNN:2102.09844:EGNN.pdf
02-Hitchhikers:2312.07511:Hitchhikers.pdf
03-MPNN:1704.01212:MPNN.pdf
05-SchNet:1706.08566:SchNet.pdf
06-TFN:1802.08219:TFN.pdf
07-SE3-Transformer:2006.10503:SE(3).pdf
08-EquiFormer:2206.11990:EquiFormer.pdf
09-Nequip:2101.03164:Nequip.pdf
10-ManifoldFormer:2511.16828:ManifoldFormer.pdf
"

only="${1:-}"
for entry in $PAPERS; do
    folder="${entry%%:*}"
    rest="${entry#*:}"
    id="${rest%%:*}"
    pdf="${rest#*:}"
    [ -n "$only" ] && [ "$folder" != "$only" ] && continue
    dest="$folder/$pdf"
    if [ -f "$dest" ]; then
        echo "skip  $dest (exists)"
        continue
    fi
    mkdir -p "$folder"
    echo "fetch $dest  (arXiv:$id)"
    curl -fsSL "https://arxiv.org/pdf/$id" -o "$dest"
done
