#!/usr/bin/env bash
# Shared environment, GCS REST helper, and source packaging utilities for F-GPO Colab operations.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$HOME/google-cloud-sdk/bin:$PATH"

COLAB_CLI="${COLAB_CLI:-$HOME/.local/bin/colab}"
COLAB_PY="${COLAB_PY:-$HOME/.local/share/uv/tools/google-colab-cli/bin/python}"
ADC_PATH="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}"
FGPO_GCS_BUCKET="${FGPO_GCS_BUCKET:-r032-pact-rebuild-f1be230a}"
FGPO_GCS_PREFIX="${FGPO_GCS_PREFIX:-fgpo-r3}"

if [ -z "${HTTPS_PROXY:-}" ] && [ -z "${https_proxy:-}" ]; then
  if command -v python3 >/dev/null 2>&1 && python3 -c "import socket; s=socket.create_connection(('127.0.0.1',7897),0.2); s.close()" 2>/dev/null; then
    export HTTPS_PROXY="http://127.0.0.1:7897"
    export HTTP_PROXY="http://127.0.0.1:7897"
  fi
fi

colab_ssh_run() {
  local session="$1"
  shift
  ssh -o ProxyCommand="$COLAB_CLI ssh --proxy-mode -s $session" \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR \
      root@colab-runtime "$@"
}

pack_fgpo_source() {
  local out_tar="$1"
  COPYFILE_DISABLE=1 tar \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='._*' \
    -czf "$out_tar" -C "$REPO_ROOT" src pyproject.toml configs scripts
}

gcs_py() {
  "$COLAB_PY" - "$ADC_PATH" "$FGPO_GCS_BUCKET" "$@" << 'PY'
import json, os, sys, time, urllib.parse, requests

adc_path, bucket, action = sys.argv[1:4]
args = sys.argv[4:]
creds = json.load(open(adc_path))
token_resp = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id": creds["client_id"],
        "client_secret": creds["client_secret"],
        "refresh_token": creds["refresh_token"],
        "grant_type": "refresh_token",
    },
    timeout=20,
)
token_resp.raise_for_status()
token = token_resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

if action == "upload":
    local_path, obj_name = args[0], args[1]
    ctype = args[2] if len(args) > 2 else "application/octet-stream"
    t0 = time.time()
    with open(local_path, "rb") as f:
        r = requests.post(
            f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o",
            params={"uploadType": "media", "name": obj_name},
            headers={**headers, "Content-Type": ctype},
            data=f,
            timeout=600,
        )
    r.raise_for_status()
    size_kb = os.path.getsize(local_path) / 1024.0
    print(f"[GCS] Uploaded gs://{bucket}/{obj_name} ({size_kb:.1f} KB) in {time.time()-t0:.2f}s")
elif action == "download":
    obj_name, local_path = args[0], args[1]
    enc = urllib.parse.quote(obj_name, safe="")
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    with requests.get(
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{enc}?alt=media",
        headers=headers,
        stream=True,
        timeout=600,
    ) as r:
        if r.status_code == 404:
            sys.exit(44)
        r.raise_for_status()
        tmp = local_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp, local_path)
elif action == "stat":
    obj_name = args[0]
    enc = urllib.parse.quote(obj_name, safe="")
    r = requests.get(
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{enc}",
        headers=headers,
        timeout=20,
    )
    if r.status_code == 404:
        sys.exit(44)
    r.raise_for_status()
    meta = r.json()
    print(json.dumps({"name": meta.get("name"), "size": meta.get("size"), "updated": meta.get("updated")}))
elif action == "list":
    prefix = args[0]
    page_token = None
    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token
        r = requests.get(
            f"https://storage.googleapis.com/storage/v1/b/{bucket}/o",
            headers=headers,
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        for item in payload.get("items", []):
            print(f"{item['name']}\t{item.get('size', 0)}\t{item.get('updated', '')}")
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
else:
    raise SystemExit(f"Unknown gcs_py action: {action}")
PY
}
