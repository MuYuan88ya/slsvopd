#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "$0")/../.." && pwd)"
ssh_target="${KAGGLE_SSH_TARGET:-kaggle}"
remote_project_dir="${KAGGLE_PROJECT_DIR:-/root/Vision-OPD}"

# Leading slashes anchor generated-directory exclusions at the project root.
# An unanchored `data/` rule would also remove source such as
# verl/trainer/config/data/ and produce an incomplete remote installation.
rsync -az --partial --info=progress2 \
    --exclude='/.git/' \
    --exclude='**/__pycache__/' \
    --exclude='*.pyc' \
    --exclude='/.venv/' \
    --exclude='/checkpoints/' \
    --exclude='/rollouts/' \
    --exclude='/data/' \
    --exclude='/outputs/' \
    --exclude='*.safetensors' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='.env' \
    --exclude='*.key' \
    --exclude='*.pem' \
    "${project_root%/}/" \
    "${ssh_target}:${remote_project_dir%/}/"

