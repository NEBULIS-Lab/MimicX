#!/usr/bin/env python3
"""Headless GVHMR-to-GMR robot retargeting."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GMR_ROOT = Path(os.environ.get("MIMICX_GMR_ROOT", PROJECT_ROOT / "third_party/GMR")).expanduser()
if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))



def load_gvhmr_pred_file_batched(gvhmr_pred_file: Path, smplx_body_model_path: Path):
    """Load GVHMR output with explicit SMPL-X batch dimensions.

    GMR's upstream loader keeps betas as a single row. Some `smplx` versions do
    not broadcast that against per-frame pose/expression tensors, so this helper
    expands shape and expression tensors to the video length.
    """

    import smplx
    import torch

    gvhmr_pred = torch.load(gvhmr_pred_file, map_location="cpu", weights_only=False)
    smpl_params_global = gvhmr_pred["smpl_params_global"]
    num_frames = smpl_params_global["body_pose"].shape[0]
    betas = smpl_params_global["betas"][0].numpy()

    smplx_data = {
        "pose_body": smpl_params_global["body_pose"].numpy(),
        "betas": betas,
        "root_orient": smpl_params_global["global_orient"].numpy(),
        "trans": smpl_params_global["transl"].numpy(),
        "mocap_frame_rate": torch.tensor(30),
    }
    body_model = smplx.create(
        str(smplx_body_model_path),
        "smplx",
        gender="neutral",
        use_pca=False,
    )
    smplx_output = body_model(
        betas=torch.tensor(np.repeat(betas[None, :], num_frames, axis=0)).float(),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )
    actual_human_height = 1.66 + 0.1 * betas[0]
    return smplx_data, body_model, smplx_output, actual_human_height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="GVHMR hmr4d_results.pt")
    parser.add_argument("--output", required=True, type=Path, help="Output GMR robot motion .pkl")
    parser.add_argument("--robot", default="unitree_g1", help="GMR target robot name")
    parser.add_argument("--fps", default=30, type=int, help="Target retargeting FPS")
    parser.add_argument(
        "--max-frames",
        default=None,
        type=int,
        help="Optional frame cap for quick smoke tests.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--body-models", type=Path, default=GMR_ROOT / "assets/body_models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast

    smplx_root = args.body_models.expanduser().resolve()

    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file_batched(
        args.input, smplx_root
    )
    frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.fps
    )
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
        verbose=not args.quiet,
    )

    qpos_list: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        qpos = retarget.retarget(frame)
        qpos_list.append(qpos)
        if not args.quiet and (index == 0 or (index + 1) % 50 == 0 or index + 1 == len(frames)):
            print(f"retargeted {index + 1}/{len(frames)} frames")

    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
        "source": str(args.input),
        "robot": args.robot,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(motion_data, f)

    print(
        f"saved {args.output} frames={len(qpos_list)} fps={aligned_fps} "
        f"dof_dim={dof_pos.shape[1] if dof_pos.ndim == 2 else 0}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
