#!/usr/bin/env bash
# Run on the assigned server after the complete source delivery is authorized.
set -eu
site_id=$1
source_root=$2
shift 2
case "$site_id" in
  host_a)
    project_root=/apps/users/adminuser/prism-exact-reproduction
    native_python="$project_root/.venv-prism-native/bin/python"
    export LD_PRELOAD=/apps/users/adminuser/miniconda3/envs/chronomop/lib/libstdc++.so.6
    ;;
  host_b)
    project_root=/home/chenxiangwei/workspace/long-horizon-decision
    native_python=/home/chenxiangwei/env/prism-native-v1/bin/python
    ;;
esac
batch_root="$project_root/outputs/training_diagnostic/native-typed-short16-20261106"
mkdir -p "$batch_root"
mkdir "$batch_root/launch-$site_id"
cd "$project_root"
nohup "$native_python" "$source_root/scripts/run_prism_training_diagnostic.py" \
  --config "$source_root/configs/prism-native-${site_id//_/-}.json" \
  --batch-plan "$source_root/configs/prism-native-short-training-20261106.json" \
  --source-root "$source_root" --output-dir "$batch_root/$site_id" \
  --gpus "$@" > "$batch_root/launch-$site_id/process.log" 2>&1 < /dev/null &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > "$batch_root/launch-$site_id/pid"
printf 'site=%s pid=%s output=%s\n' "$site_id" "$launcher_pid" "$batch_root/$site_id"
