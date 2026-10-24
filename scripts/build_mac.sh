#!/bin/zsh
# Build the native Apple Silicon app locally: dist/RS-StatChem.app and a zip next to it.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
$PY -m PyInstaller packaging/rsstatchem.spec --noconfirm --clean
VERSION=$($PY -c "import rsstatchem; print(rsstatchem.__version__)")
ditto -c -k --keepParent dist/RS-StatChem.app "dist/RS-StatChem-$VERSION-macOS-arm64.zip"
echo "Built dist/RS-StatChem.app ($(lipo -archs dist/RS-StatChem.app/Contents/MacOS/RS-StatChem))"
