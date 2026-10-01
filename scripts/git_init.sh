#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
[ -d .git ] || git init
git add -A
git commit -m "UR5e + MoveIt2 + colour vision + action skills + Ollama LLM bridge (working pipeline)" || true
git tag -f v0.1-pipeline-works
git log --oneline | head -3
