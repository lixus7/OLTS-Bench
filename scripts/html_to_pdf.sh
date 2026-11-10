#!/usr/bin/env bash
# Print an HTML report to an A4 PDF with a headless Chrome or Chromium.
# Usage: bash scripts/html_to_pdf.sh report.html [out.pdf]
set -euo pipefail
in=$(realpath "$1"); out=${2:-${1%.html}.pdf}
for b in google-chrome chromium chromium-browser chrome; do
  if command -v "$b" >/dev/null; then browser=$b; break; fi
done
if [ -z "${browser:-}" ]; then
  echo "no Chrome/Chromium found: open $1 in a browser and print to PDF (A4, background graphics on)"; exit 1
fi
"$browser" --headless --no-sandbox --disable-gpu --no-pdf-header-footer --virtual-time-budget=10000 --print-to-pdf="$out" "file://$in" 2>/dev/null
echo "wrote $out"
