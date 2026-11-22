#!/usr/bin/env bash
set -euo pipefail

archive=${1:?Pass the locally prepared BiPACE TextCraft archive path}
patch_file=${2:-/content/tools/recipe-traversal-replay.patch}
data_parent=${R5_DATA_PARENT:-/content/data}
venv=${R5_TEXTCRAFT_VENV:-/content/venvs/textcraft}
main_python=${R5_MAIN_PYTHON:-/content/venvs/fgpo-main/bin/python}
wheels="$data_parent/textcraft-bipace-f19f6fe/wheels"

mkdir -p "$data_parent" "$(dirname -- "$venv")"
tar --no-same-owner -xzf "$archive" -C "$data_parent"
"$main_python" -m venv --without-pip "$venv"
export PIP_CONFIG_FILE=/dev/null PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1
PYTHONPATH="$wheels/pip-25.3-py3-none-any.whl" "$venv/bin/python" -m pip install \
    --ignore-installed --no-index --find-links "$wheels" pip==25.3 setuptools==80.9.0 wheel==0.45.1
"$venv/bin/python" -m pip install --no-index --find-links "$wheels" --no-build-isolation \
    numpy==1.25.2 ray==2.53.0 gym==0.26.2 gymnasium==0.29.1

crafting_tree="$data_parent/textcraft-bipace-f19f6fe/vendor/agent_system/environments/env_package/textcraft/textcraft_src/crafting_tree.py"
if ! grep -q 'recipes = list(' "$crafting_tree"; then
    if command -v patch >/dev/null 2>&1; then
        patch -d "$data_parent/textcraft-bipace-f19f6fe/vendor" -p1 < "$patch_file"
    else
        "$main_python" - "$crafting_tree" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
old = "recipes = self.itemid_recipes.get(item_name) or self.tag_recipes.get(item_name) or []"
new = "recipes = list(self.itemid_recipes.get(item_name) or self.tag_recipes.get(item_name) or [])"
if old not in src:
    raise SystemExit(f"Expected target line not found in {p}")
p.write_text(src.replace(old, new, 1), encoding="utf-8")
PY
    fi
fi
grep -q 'recipes = list(' "$crafting_tree"
