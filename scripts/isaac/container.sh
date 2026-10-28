#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo 'Usage: MIMICX_ISAAC_RUNTIME=/own/runtime container.sh [--cpu | --gpu GPU-UUID-OR-INDEX] [--shell] command...'
}
ROOT=${MIMICX_ISAAC_RUNTIME:?Set MIMICX_ISAAC_RUNTIME to a project-owned runtime directory}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CODE=${MIMICX_ISAAC_CODE:-$CODE}
IMAGE=nvcr.io/nvidia/isaac-sim@sha256:68735a60b6c15c85e0dd0098570c6d2cc79e928f2d068ce2790aa43284ac165d
mode=cpu
gpu=
shell=0
while (($#)); do
    case "$1" in
        --cpu) mode=cpu; shift ;;
        --gpu) mode=gpu; gpu=${2:?GPU UUID required}; shift 2 ;;
        --shell) shell=1; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
(($#)) || { usage >&2; exit 2; }
[[ -d "$ROOT/sources/IsaacLab/source" ]] || { echo 'Prepare pinned IsaacLab sources first' >&2; exit 2; }
mkdir -p "$ROOT"/{env,cache/kit,home,logs/kit,outputs,assets}
args=(run --rm --init --user 0:0 --entrypoint /bin/bash
    --shm-size=2g --ulimit nofile=65535:65535 --network host
    -e http_proxy -e https_proxy -e HTTP_PROXY -e HTTPS_PROXY -e ALL_PROXY -e NO_PROXY
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e HOME=/runtime/home
    -e HUB__ARGS__DETECT_ONLY=false -e HUB__CACHE__PATH=/runtime/cache/hub
    -e "GIT_PYTHON_REFRESH=${GIT_PYTHON_REFRESH:-quiet}"
    -e OMNI_KIT_ALLOW_ROOT=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1
    -e XDG_CACHE_HOME=/runtime/cache -e PIP_CACHE_DIR=/runtime/cache/pip
    -e ISAACSIM_PATH=/isaac-sim -e ISAACLAB_PATH=/runtime/sources/IsaacLab
    --mount "type=bind,src=$ROOT,dst=/runtime"
    --mount "type=bind,src=$ROOT/cache/kit,dst=/isaac-sim/kit/cache"
    --mount "type=bind,src=$ROOT/logs/kit,dst=/isaac-sim/.nvidia-omniverse/logs"
    --mount "type=bind,src=$CODE,dst=/workspace,readonly"
    -w /workspace)
if [[ $mode == cpu ]]; then
    args+=(--runtime=runc -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=)
else
    [[ $gpu =~ ^GPU-[0-9a-fA-F-]+$ || $gpu =~ ^[0-9]+$ ]] || { echo 'Use one explicit GPU UUID or index' >&2; exit 2; }
    args+=(--runtime=nvidia --gpus "device=$gpu" -e "NVIDIA_VISIBLE_DEVICES=$gpu"
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display -e CUDA_VISIBLE_DEVICES=0)
fi
printf 'pid=%s mode=%s gpu=%s image=%s command=' "$$" "$mode" "${gpu:-none}" "$IMAGE" >&2
printf '%q ' "$@" >&2
printf '\n' >&2
exec docker "${args[@]}" "$IMAGE" -c '
    set -eo pipefail
    umask 0002
    export CARB_APP_PATH=/isaac-sim/kit ISAAC_PATH=/isaac-sim EXP_PATH=/isaac-sim/apps
    export LD_PRELOAD=/isaac-sim/kit/libcarb.so
    export RESOURCE_NAME=IsaacSim
    export PATH=/runtime/tools/bin:$PATH
    source /isaac-sim/setup_python_env.sh
    export PYTHONPATH=/runtime/env/venv/lib/python3.12/site-packages:/runtime/env/site-packages:/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle:$PYTHONPATH
    for pkg in /runtime/sources/IsaacLab/source/*; do
        [[ ! -f "$pkg/setup.py" ]] || export PYTHONPATH="$PYTHONPATH:$pkg"
    done
    export PYTHONPATH="$PYTHONPATH:/runtime/sources/whole_body_tracking/source/whole_body_tracking:/workspace"
    export LD_LIBRARY_PATH="/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/torch/lib:$LD_LIBRARY_PATH"
    if [[ "$1" == 1 ]]; then shift; exec "$@"; fi
    shift
    python=/isaac-sim/kit/python/bin/python3
    [[ ! -x /runtime/env/venv/bin/python ]] || python=/runtime/env/venv/bin/python
    exec "$python" "$@"
' bash "$shell" "$@"
