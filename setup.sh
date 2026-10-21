#!/bin/bash
# Linux / macOS: Python 3.12 venv + the competition's released v0.2.0 simulator (public repo, tag v0.2.0 = 62859c2).
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
if [ ! -d apex-competition-nist-assembly ]; then
  git clone https://github.com/unarbos/apex-competition-nist-assembly.git ||
    git clone https://github.com/macrocosm-os/apex-competition-nist-assembly.git
fi
git -C apex-competition-nist-assembly fetch --all --tags -q || true
git -C apex-competition-nist-assembly checkout -q 62859c2
echo "Setup done. Start with ./run.sh and open http://127.0.0.1:8765"
