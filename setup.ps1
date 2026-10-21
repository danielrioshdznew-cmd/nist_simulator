# Windows (PowerShell): Python 3.12 venv + the competition's released v0.2.0 simulator (public repo, tag v0.2.0 = 62859c2).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
if (!(Test-Path apex-competition-nist-assembly)) {
  git clone https://github.com/unarbos/apex-competition-nist-assembly.git
  if ($LASTEXITCODE -ne 0) { git clone https://github.com/macrocosm-os/apex-competition-nist-assembly.git }
}
git -C apex-competition-nist-assembly fetch --all --tags -q
git -C apex-competition-nist-assembly checkout -q 62859c2
Write-Host "Setup done. Start with .\run.ps1 and open http://127.0.0.1:8765"
