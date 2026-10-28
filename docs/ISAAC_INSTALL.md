# Install the Experimental Isaac Backend

This is the isolated Linux x86_64 container setup for the **tested Sim 6.0 /
Isaac Lab v3.0.0-beta2 / RSL-RL 5.0.1** port. It does not modify a MuJoCo
environment. The source of truth is
[`isaaclab-sources.json`](../dependencies/isaaclab-sources.json), with Python
packages in [`isaaclab.txt`](../dependencies/isaaclab.txt) and the public launcher
[`container.sh`](../scripts/isaac/container.sh). Do not substitute Lab main,
another simulator image, or a newer RSL-RL release.

The CPU install sequence below matches the one used for the runtime smoke tests.
The public bootstrap obtains the same pinned sources from upstream instead of
copying another workstation's source tree. No private scripts, model cache,
machine-specific git binary, or existing Isaac installation are required.

## Prerequisites and Terms

- Host tools: Bash, Python 3, git, curl, tar, sha256sum, Docker, and access to
  the pinned NVIDIA container image. Host git is used to obtain and identify
  sources; **git inside the container is optional**.
- Physical training/rollout requires a compatible NVIDIA RTX GPU, host driver,
  and NVIDIA Container Toolkit. CPU installation/tests do not start the
  simulator or validate GPU compatibility. Review NVIDIA's
  [Sim 6 container instructions](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_container.html)
  and [hardware requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html).
- Review the image's NVIDIA license and data-collection terms **before** pulling
  or running it. The launcher currently passes `ACCEPT_EULA=Y` and
  `PRIVACY_CONSENT=Y`; invoking it opts into those settings. Do not invoke it
  unless you agree. NVIDIA documents both flags and the privacy opt-out in its
  [container environment-variable documentation](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_container.html#container-deployment).
- Preserve the upstream licenses for Isaac Lab and `whole_body_tracking`.
  The separately acquired Unitree asset archive is not covered simply by
  MimicX's or WBT's code license. The audited archive had no LICENSE-named file;
  confirm your permitted use and redistribution rights separately. Do not
  bundle the simulator, robot archive, or licensed motion datasets into a code
  release. Supply your own authorized motion data.

The launcher uses host networking, runs as container root, mounts only the
chosen runtime writable, and mounts the MimicX checkout read-only. Use a
dedicated trusted runtime directory; do not expose unrelated data. Some Kit
assets still use network/cache lookups, so this is not an offline-install claim.

## 1. Clean Checkout and Runtime Layout

Start from a MimicX release/commit containing this document and record its
revision. The runtime directory below must be new; choose a different name
instead of reusing an active environment or run.

```bash
set -euo pipefail
git clone https://github.com/NEBULIS-Lab/MimicX.git
cd MimicX
export MIMICX_CODE_ROOT="$PWD"
export MIMICX_ISAAC_RUNTIME="$HOME/mimicx-isaac-runtime"
test ! -e "$MIMICX_ISAAC_RUNTIME"
mkdir -p "$MIMICX_ISAAC_RUNTIME"/{sources,downloads,logs,outputs,assets}
```

All following commands run from this checkout, in the same shell. Inside the
container, the checkout is `/workspace` and this runtime is `/runtime`:

```text
runtime/
  sources/IsaacLab/                   pinned source checkout
  sources/IsaacLab/_isaac_sim         symlink to /isaac-sim inside the image
  sources/whole_body_tracking/        pinned source checkout plus robot assets
  env/venv/                          private Python 3.12 environment
  downloads/                         verified robot archive
  logs/                              source IDs, package freeze, checks
  assets/                            your authorized named motion inputs
  outputs/                           new checkpoints, metrics and trajectories
  cache/ and home/                   created by container.sh
```

## 2. Pinned Sources and Robot Assets

Read identities from the checked-in manifest. The GitHub repositories may
require network access or ordinary authentication; no W&B registry is used.

```bash
export LAB_REPO=$(python3 -c 'import json; print(json.load(open("dependencies/isaaclab-sources.json"))["isaac_lab"]["repository"])')
export LAB_COMMIT=$(python3 -c 'import json; print(json.load(open("dependencies/isaaclab-sources.json"))["isaac_lab"]["commit"])')
export WBT_REPO=$(python3 -c 'import json; print(json.load(open("dependencies/isaaclab-sources.json"))["whole_body_tracking"]["repository"])')
export WBT_COMMIT=$(python3 -c 'import json; print(json.load(open("dependencies/isaaclab-sources.json"))["whole_body_tracking"]["commit"])')
export ROBOT_URL=$(python3 -c 'import json; print(json.load(open("dependencies/isaaclab-sources.json"))["unitree_description"]["url"])')
export ROBOT_SHA=$(python3 -c 'import json; print(json.load(open("dependencies/isaaclab-sources.json"))["unitree_description"]["archive_sha256"])')
export ISAAC_IMAGE="nvcr.io/nvidia/isaac-sim@$(python3 -c 'import json; print(json.load(open("dependencies/isaaclab-sources.json"))["isaac_sim"]["digest"])')"

git clone --filter=blob:none --no-checkout "$LAB_REPO" "$MIMICX_ISAAC_RUNTIME/sources/IsaacLab"
git -C "$MIMICX_ISAAC_RUNTIME/sources/IsaacLab" checkout --detach "$LAB_COMMIT"
git clone --filter=blob:none --no-checkout "$WBT_REPO" "$MIMICX_ISAAC_RUNTIME/sources/whole_body_tracking"
git -C "$MIMICX_ISAAC_RUNTIME/sources/whole_body_tracking" checkout --detach "$WBT_COMMIT"

curl --fail --location --retry 4 --connect-timeout 30 \
  "$ROBOT_URL" -o "$MIMICX_ISAAC_RUNTIME/downloads/unitree_description.tar.gz"
printf '%s  %s\n' "$ROBOT_SHA" "$MIMICX_ISAAC_RUNTIME/downloads/unitree_description.tar.gz" | sha256sum --check -
export WBT_ASSETS="$MIMICX_ISAAC_RUNTIME/sources/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets"
tar --extract --gzip --file "$MIMICX_ISAAC_RUNTIME/downloads/unitree_description.tar.gz" --directory "$WBT_ASSETS"
test -f "$WBT_ASSETS/unitree_description/urdf/g1/main.urdf"

ln -s /isaac-sim "$MIMICX_ISAAC_RUNTIME/sources/IsaacLab/_isaac_sim"
docker pull "$ISAAC_IMAGE"
```

The `_isaac_sim` symlink deliberately resolves only inside the container. The
robot archive must extract to the exact path above; do not strip its
`unitree_description/` top-level directory. No shared simulator image or upstream
source patches are needed: compatibility changes live in MimicX's adapter.

Record checked-out source identities separately from optional container git
logging. This also verifies that the two upstream HEADs match the manifest:

```bash
python3 - <<'PY'
import hashlib
import json
import os
import subprocess
from pathlib import Path

code = Path(os.environ["MIMICX_CODE_ROOT"])
root = Path(os.environ["MIMICX_ISAAC_RUNTIME"])
manifest = code / "dependencies/isaaclab-sources.json"
pins = json.loads(manifest.read_text())
def head(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
lab = head(root / "sources/IsaacLab")
wbt = head(root / "sources/whole_body_tracking")
assert lab == pins["isaac_lab"]["commit"]
assert wbt == pins["whole_body_tracking"]["commit"]
receipt = {"mimicx_commit": head(code), "isaac_lab_commit": lab,
           "whole_body_tracking_commit": wbt,
           "image": os.environ["ISAAC_IMAGE"],
           "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
           "robot_archive_sha256": pins["unitree_description"]["archive_sha256"]}
(root / "logs/source-identities.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
PY
```

## 3. CPU-Only Environment Installation

The launcher defaults to CPU isolation (`runc`, no NVIDIA devices). The pinned
image supplies Python 3.12.13, Torch 2.11.0+cu128 and torchvision 0.26.0+cu128;
the launcher exposes its ML prebundle. Do not replace them with host packages
or run the following pip commands in the MuJoCo environment.

```bash
bash scripts/isaac/container.sh --cpu --shell bash -c '
set -euo pipefail
test -z "$(find /dev -name "nvidia*" -print -quit)"
/isaac-sim/kit/python/bin/python3 -m venv --system-site-packages /runtime/env/venv
PY=/runtime/env/venv/bin/python
"$PY" -m pip install -r /workspace/dependencies/isaaclab.txt
for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic isaaclab_ov isaaclab_ovphysx; do
    "$PY" -m pip install --no-deps --no-build-isolation -e "/runtime/sources/IsaacLab/source/$pkg"
done
"$PY" -m pip install --no-deps --no-build-isolation -e /runtime/sources/whole_body_tracking/source/whole_body_tracking
"$PY" -m pip freeze > /runtime/logs/installed-freeze.txt
set +e
"$PY" -m pip check > /runtime/logs/pip-check.txt 2>&1
check_status=$?
set -e
printf "pip check status=%s\n" "$check_status"
cat /runtime/logs/pip-check.txt
"$PY" -c "import sys,torch,numpy,rsl_rl; print(sys.version,torch.__version__,numpy.__version__,rsl_rl.__file__); print(torch.ones(2,device=\"cpu\")+1); assert not torch.cuda.is_initialized()"
'
```

This is a targeted Kit/PhysX overlay, not installation of every Lab extra or
Kit-less backend. The tested environment's `pip check` reports unmet optional
and base-image declarations; preserve and review that report rather than
calling the environment globally dependency-clean. Do not install `[all]`,
another `isaacsim` distribution, or pip `usd-core` over Kit's bundled `pxr` to
silence unrelated declarations. The wrapper adds source packages and `/workspace`
to `PYTHONPATH`; no editable install into the read-only MimicX mount is needed.

## 4. Validate Without Container Git or GPU

The image does not provide git. No host binary needs to be copied into it.
`GIT_PYTHON_REFRESH=quiet` lets GitPython import without git; RSL's best-effort
repository logging skips unavailable repositories. The public launcher and
backend default this variable to `quiet`, while respecting an explicit override.
Actual source identities are retained in the separate receipt above and the
backend's declared-source manifest metadata.

The following temporary PATH excludes any custom tools directory. It was
tested in a CPU container: git was absent, RSL imports and the logger's
repository-absence path passed, and CUDA remained uninitialized.

```bash
bash scripts/isaac/container.sh --cpu --shell env \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  GIT_PYTHON_REFRESH=quiet /runtime/env/venv/bin/python -c '
import os
import shutil
from pathlib import Path
import torch
import git
import rsl_rl
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger
assert shutil.which("git") is None
assert not list(Path("/dev").glob("nvidia*"))
assert os.environ["GIT_PYTHON_REFRESH"] == "quiet"
logger = Logger.__new__(Logger)
logger.log_dir = "/runtime/outputs/cpu-no-git-check"
logger.disable_logs = False
logger.git_status_repos = [rsl_rl.__file__]
assert logger._store_code_state() == []
assert not torch.cuda.is_initialized()
print("CPU_NO_GIT_IMPORT_AND_LOGGER_PASS")
'
bash scripts/isaac/container.sh --cpu -m pytest tests/isaac tests/autorefine tests/runtime -q
```

These CPU checks do not construct `SimulationApp`. A real GPU run remains a
separate gate. See [ISAAC.md](ISAAC.md) for named motion conversion, exact task
and checkpoint contracts, telemetry clocks, and the validated smoke scope.

## 5. Run the Backend

Put a converted, authorized named motion in `assets/motion.npz`. Select exactly
one compatible GPU already available to you, then run the public entry point:

```bash
bash scripts/isaac/container.sh --gpu GPU-YOUR-UUID \
  scripts/isaac/train.py --motion-file /runtime/assets/motion.npz \
  --output-root /runtime/outputs/train --run-name initial \
  --iterations 200 --num-envs 1024 --headless
```

`GPU-YOUR-UUID` is a placeholder for an actual UUID; a single device index is
also accepted. The launcher restricts device visibility; it is not a resource
scheduler. Check runtime completion artifacts, not only the process exit code.
Use a new run name for continuation. Checkpoints from MuJoCo or another RSL
major version are not implicitly compatible.

The container writes as root, so atomic state files may be unreadable to the
host user. After a run has ended, a narrowly scoped ownership repair for only
that run directory can be performed through a CPU container:

```bash
bash scripts/isaac/container.sh --cpu --shell chown -R "$(id -u):$(id -g)" \
  /runtime/outputs/train/initial
```

Do not apply broad permission changes to shared storage or another run. Keep
package-freeze/source receipts with the run; no git executable is needed for
policy training, checkpoint loading, rollout, or AutoRefine itself.
