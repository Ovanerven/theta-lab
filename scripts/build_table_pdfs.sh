#!/usr/bin/env bash
# Compile the generated appendix tables into preview PDFs you can eyeball.
# Output: results/sweep_consolidated/tables/preview_<layout>_<metric>.pdf
# Each appendix_*.tex is a \input-able fragment; here we wrap it in a minimal
# standalone document and compile from its own folder (so bare \input works).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TBL="$ROOT/results/sweep_consolidated/tables"

build() {  # $1=layout (per_scaffold|per_system)  $2=metric  $3=orientation note
  local layout="$1" metric="$2"
  local dir="$TBL/$layout/$metric"
  local frag="appendix_${layout}_${metric}.tex"
  [ -f "$dir/$frag" ] || { echo "skip: $dir/$frag missing"; return; }
  local master="$dir/_preview_master.tex"
  cat > "$master" <<EOF
\\documentclass[10pt]{article}
\\usepackage[a4paper,margin=2.5cm]{geometry}
\\usepackage{booktabs}
\\usepackage{caption}
\\usepackage{graphicx}
\\setlength{\\tabcolsep}{4pt}
\\begin{document}
\\input{$frag}
\\end{document}
EOF
  ( cd "$dir" && pdflatex -interaction=nonstopmode -halt-on-error _preview_master.tex >/tmp/tbl_$$.log 2>&1 \
      && pdflatex -interaction=nonstopmode -halt-on-error _preview_master.tex >/tmp/tbl_$$.log 2>&1 ) \
    || { echo "FAILED $layout/$metric — see /tmp/tbl_$$.log"; tail -15 /tmp/tbl_$$.log; return; }
  mv "$dir/_preview_master.pdf" "$TBL/preview_${layout}_${metric}.pdf"
  rm -f "$dir"/_preview_master.{tex,aux,log,out}
  local of; of=$(grep -c 'Overfull \\hbox' /tmp/tbl_$$.log || true)
  echo "OK  preview_${layout}_${metric}.pdf  (overfull boxes: ${of:-0})"
}

build per_scaffold median
build per_scaffold mean
build per_system  median
build per_system  mean
echo "PDFs in: $TBL/"
