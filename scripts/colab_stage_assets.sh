#!/usr/bin/env bash
# Stage F-GPO r3 cold assets (venvs, Qwen2.5-7B-Instruct, ALFWorld/AppWorld data)
# onto GCS using a 0-GPU CPU-only Colab VM so billed A100 sessions never wait on pip/HF downloads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/colab_common.sh"

STAGE_SESSION="fgpo-stage-cpu"
APPWORLD_BUNDLE_OBJ=""
SKIP_APPWORLD="0"
FORCE="0"

while [ $# -gt 0 ]; do
  case "$1" in
    -s|--session) STAGE_SESSION="$2"; shift 2 ;;
    --bucket) FGPO_GCS_BUCKET="$2"; shift 2 ;;
    --prefix) FGPO_GCS_PREFIX="$2"; shift 2 ;;
    --appworld-bundle-obj) APPWORLD_BUNDLE_OBJ="$2"; shift 2 ;;
    --skip-appworld) SKIP_APPWORLD="1"; shift ;;
    --force) FORCE="1"; shift ;;
    -h|--help)
      echo "Usage: $0 [-s fgpo-stage-cpu] [--bucket BUCKET] [--prefix fgpo-r3] [--appworld-bundle-obj OBJ] [--skip-appworld] [--force]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cleanup() {
  echo "[stage] Releasing CPU staging session '$STAGE_SESSION'..."
  "$COLAB_CLI" stop -s "$STAGE_SESSION" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[stage] Creating 0-GPU CPU session '$STAGE_SESSION'..."
"$COLAB_CLI" new -s "$STAGE_SESSION" --high-mem || "$COLAB_CLI" new -s "$STAGE_SESSION"
"$COLAB_CLI" upload -s "$STAGE_SESSION" "$ADC_PATH" /root/.gcs_adc.json

colab_ssh_run "$STAGE_SESSION" \
  "chmod 600 /root/.gcs_adc.json && BUCKET='$FGPO_GCS_BUCKET' PREFIX='$FGPO_GCS_PREFIX' APPWORLD_BUNDLE_OBJ='$APPWORLD_BUNDLE_OBJ' SKIP_APPWORLD='$SKIP_APPWORLD' FORCE='$FORCE' bash -s" << 'EOF'
set -euo pipefail

apt-get update -qq && apt-get install -y -qq zstd >/dev/null

if ! command -v uv >/dev/null 2>&1; then
  if [ -x /usr/local/bin/uv ]; then
    export PATH="/usr/local/bin:$PATH"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

export UV_PYTHON_INSTALL_DIR=/content/venvs/.python
mkdir -p /content/venvs/.python /content/models /content/data /tmp/stage_pkgs

echo "[stage] CPU VM resources:"
free -g
df -BG /content

get_token() {
  python3 -c '
import json, urllib.request, urllib.parse
c = json.load(open("/root/.gcs_adc.json"))
data = urllib.parse.urlencode({
    "client_id": c["client_id"],
    "client_secret": c["client_secret"],
    "refresh_token": c["refresh_token"],
    "grant_type": "refresh_token",
}).encode()
req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data)
print(json.loads(urllib.request.urlopen(req, timeout=20).read())["access_token"])
'
}

gcs_exists() {
  local remote_obj="$1"
  local token enc code
  token="$(get_token)"
  enc="$(python3 -c 'import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$remote_obj")"
  code="$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $token" \
    "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${enc}")"
  [ "$code" = "200" ]
}

upload_gcs() {
  local local_file="$1"
  local remote_obj="$2"
  local token enc
  token="$(get_token)"
  enc="$(python3 -c 'import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$remote_obj")"
  curl -fsSL -X POST -T "$local_file" \
    -H "Authorization: Bearer $token" \
    -H "Content-Type: application/zstd" \
    "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&name=${enc}" >/dev/null
  echo "[stage] Uploaded gs://${BUCKET}/${remote_obj} ($(du -sh "$local_file" | cut -f1))"
}

NEED_MAIN="0"
NEED_APPWORLD="0"
NEED_ALF_VENV="0"
NEED_ALF_DATA="0"
NEED_QWEN="0"
if [ "$FORCE" = "1" ] || ! gcs_exists "${PREFIX}/assets/fgpo_venv_main.tar.zst"; then NEED_MAIN="1"; fi
if [ "$SKIP_APPWORLD" = "0" ] && [ -n "$APPWORLD_BUNDLE_OBJ" ]; then
  if [ "$FORCE" = "1" ] || ! gcs_exists "${PREFIX}/assets/appworld_data.tar.zst"; then NEED_APPWORLD="1"; fi
fi
if [ "$FORCE" = "1" ] || ! gcs_exists "${PREFIX}/assets/fgpo_venv_alfworld.tar.zst"; then NEED_ALF_VENV="1"; fi
if [ "$FORCE" = "1" ] || ! gcs_exists "${PREFIX}/assets/alfworld_data.tar.zst"; then NEED_ALF_DATA="1"; fi
if [ "$FORCE" = "1" ] || ! gcs_exists "${PREFIX}/assets/qwen25_7b_instruct.tar.zst"; then NEED_QWEN="1"; fi

if [ "$NEED_MAIN" = "1" ] || [ "$NEED_APPWORLD" = "1" ]; then
  echo "[stage] Building main F-GPO venv (managed Python 3.11 inside /content/venvs/.python)..."
  uv python install 3.11
  uv venv --python 3.11 --python-preference only-managed /content/venvs/fgpo-main
  uv pip install --python /content/venvs/fgpo-main/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    "torch==2.7.0+cu128"
  uv pip install --python /content/venvs/fgpo-main/bin/python \
    "transformers==4.57.6" "tokenizers==0.22.2" "accelerate>=1.0.0" "safetensors>=0.4.5" \
    "huggingface-hub>=0.26.0" "numpy==1.26.4" "scipy>=1.11.0" "peft>=0.13.0" \
    "freezegun>=1.5.0,<=1.5.1" "requests>=2.28.0" "fastapi==0.119.1" "starlette==0.48.0" \
    "fastapi-login==1.10.3" "sqlmodel==0.0.24" "sqlalchemy==2.0.57" "pydantic==2.12.5" \
    "pydantic-core==2.41.5" "sqlalchemy-utils>=0.41.1" "pendulum>=3.0.0" "uvicorn>=0.27" \
    "ipython>=8.18.0" "typer>=0.16.0" "text2num>=0.3.0" "orjson>=3.10.12" \
    "pydantic-extra-types[pendulum]>=2.8.0" "inflection>=0.5.1" "email-validator>=2.0.0" \
    "polyfactory>=2.15.0" "faker>=18.0.0" "xxhash>=3.0.0" "munch>=4.0.0" "rich>=13.0.0" \
    "tqdm>=4.60.0" "python-dotenv>=1.0.1" "cryptography>=47.0.0" "python-multipart>=0.0.5" \
    "httpx>=0.20.0" "httpx2>=2.0.0" "libcst>=1.2.0" "typing-extensions>=4.12.2" \
    "pyyaml>=6.0.1" "psutil>=5.9.0" "jsonref>=1.1.0" "filelock>=3.1.0" "uvloop>=0.21.0"

  if [ "$SKIP_APPWORLD" = "0" ] && [ -n "$APPWORLD_BUNDLE_OBJ" ]; then
    echo "[stage] Restoring AppWorld package and data from gs://${BUCKET}/${APPWORLD_BUNDLE_OBJ}..."
    TOKEN="$(get_token)"
    ENC_OBJ="$(python3 -c 'import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$APPWORLD_BUNDLE_OBJ")"
    curl -fsSL -H "Authorization: Bearer $TOKEN" \
      "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${ENC_OBJ}?alt=media" \
      -o /tmp/appworld_bundle.tar.gz
    mkdir -p /tmp/aw_extract /content/data
    tar -xzf /tmp/appworld_bundle.tar.gz -C /tmp/aw_extract
    cp -rf /tmp/aw_extract/appworld_native/appworld /content/venvs/fgpo-main/lib/python3.11/site-packages/
    cp -rf /tmp/aw_extract/appworld_native/appworld-*.dist-info /content/venvs/fgpo-main/lib/python3.11/site-packages/
    rm -rf /content/data/appworld-root
    mv /tmp/aw_extract/appworld_root /content/data/appworld-root
    rm -rf /tmp/aw_extract /tmp/appworld_bundle.tar.gz
    if [ "$NEED_APPWORLD" = "1" ]; then
      tar -I 'zstd -T0 -3' -cf /tmp/stage_pkgs/appworld_data.tar.zst -C /content/data appworld-root
      upload_gcs /tmp/stage_pkgs/appworld_data.tar.zst "${PREFIX}/assets/appworld_data.tar.zst"
      rm -f /tmp/stage_pkgs/appworld_data.tar.zst
    fi
    rm -rf /content/data/appworld-root
  fi

  /content/venvs/fgpo-main/bin/python -c 'import torch, transformers, accelerate, safetensors, huggingface_hub, numpy, peft, freezegun; print("fgpo-main imports OK", torch.__version__, transformers.__version__)'
  if [ "$SKIP_APPWORLD" = "0" ] && [ -n "$APPWORLD_BUNDLE_OBJ" ]; then
    /content/venvs/fgpo-main/bin/python -c 'import appworld; print("appworld import OK")'
  fi
  rm -rf /root/.cache/uv

  if [ "$NEED_MAIN" = "1" ]; then
    tar -I 'zstd -T0 -3' -cf /tmp/stage_pkgs/fgpo_venv_main.tar.zst -C /content/venvs .python fgpo-main
    upload_gcs /tmp/stage_pkgs/fgpo_venv_main.tar.zst "${PREFIX}/assets/fgpo_venv_main.tar.zst"
    rm -f /tmp/stage_pkgs/fgpo_venv_main.tar.zst
  fi
  rm -rf /content/venvs/fgpo-main /content/venvs/.python
else
  echo "[stage] Skip existing fgpo_venv_main.tar.zst and appworld_data.tar.zst"
fi

if [ "$NEED_ALF_VENV" = "1" ] || [ "$NEED_ALF_DATA" = "1" ]; then
  echo "[stage] Building ALFWorld worker venv / downloading data..."
  uv python install 3.11
  uv venv --python 3.11 --python-preference only-managed /content/venvs/alfworld
  uv pip install --python /content/venvs/alfworld/bin/python \
    "alfworld==0.4.2" "textworld==1.7.0" "numpy==1.26.4"
  /content/venvs/alfworld/bin/python -c 'import alfworld, textworld, numpy; print("alfworld venv imports OK")'
  if [ "$NEED_ALF_VENV" = "1" ]; then
    tar -I 'zstd -T0 -3' -cf /tmp/stage_pkgs/fgpo_venv_alfworld.tar.zst -C /content/venvs alfworld
    upload_gcs /tmp/stage_pkgs/fgpo_venv_alfworld.tar.zst "${PREFIX}/assets/fgpo_venv_alfworld.tar.zst"
    rm -f /tmp/stage_pkgs/fgpo_venv_alfworld.tar.zst
  fi
  if [ "$NEED_ALF_DATA" = "1" ]; then
    ALFWORLD_DATA=/content/data/alfworld /content/venvs/alfworld/bin/alfworld-download
    tar -I 'zstd -T0 -3' -cf /tmp/stage_pkgs/alfworld_data.tar.zst -C /content/data alfworld
    upload_gcs /tmp/stage_pkgs/alfworld_data.tar.zst "${PREFIX}/assets/alfworld_data.tar.zst"
    rm -f /tmp/stage_pkgs/alfworld_data.tar.zst
    rm -rf /content/data/alfworld
  fi
  rm -rf /root/.cache/uv /content/venvs/alfworld
else
  echo "[stage] Skip existing fgpo_venv_alfworld.tar.zst and alfworld_data.tar.zst"
fi

if [ "$NEED_QWEN" = "1" ]; then
  echo "[stage] Downloading Qwen/Qwen2.5-7B-Instruct..."
  uv venv --python 3.11 /tmp/hf_dl
  uv pip install --python /tmp/hf_dl/bin/python "huggingface-hub>=0.26.0" "hf-xet>=0.1.0" || \
    uv pip install --python /tmp/hf_dl/bin/python "huggingface-hub>=0.26.0"
  /tmp/hf_dl/bin/python -c '
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    revision="a09a35458c702b33eeacc393d103063234e8bc28",
    local_dir="/content/models/Qwen2.5-7B-Instruct",
)
'
  rm -rf /content/models/Qwen2.5-7B-Instruct/.cache /root/.cache/huggingface /root/.cache/uv /tmp/hf_dl
  tar -I 'zstd -T0 -3' -cf /tmp/stage_pkgs/qwen25_7b_instruct.tar.zst -C /content/models Qwen2.5-7B-Instruct
  upload_gcs /tmp/stage_pkgs/qwen25_7b_instruct.tar.zst "${PREFIX}/assets/qwen25_7b_instruct.tar.zst"
  rm -f /tmp/stage_pkgs/qwen25_7b_instruct.tar.zst
  rm -rf /content/models/Qwen2.5-7B-Instruct
else
  echo "[stage] Skip existing qwen25_7b_instruct.tar.zst"
fi
echo "[stage] All cold assets staged to gs://${BUCKET}/${PREFIX}/assets/."
EOF
