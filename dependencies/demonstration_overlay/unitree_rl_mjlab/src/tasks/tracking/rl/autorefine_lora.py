"""AutoRefine LoRA policy adapters for motion-specific refinement."""

from __future__ import annotations

from itertools import chain
import copy
import json
import math
import os
from typing import Any

import torch
from torch import nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_optimizer, unpad_trajectories

from .runner import MotionTrackingOnPolicyRunner


def _env_bool(name: str, default: bool = False) -> bool:
  value = os.environ.get(name)
  if value is None:
    return default
  return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
  value = os.environ.get(name)
  if value is None or value == "":
    return default
  return int(value)


def _env_float(name: str, default: float) -> float:
  value = os.environ.get(name)
  if value is None or value == "":
    return default
  return float(value)


def _parse_scalings(max_rank: int) -> torch.Tensor:
  json_value = os.environ.get("MIMICX_AUTOREFINE_LORA_BLOCK_SCALINGS")
  if json_value:
    values = json.loads(json_value)
    if not isinstance(values, list):
      raise TypeError("MIMICX_AUTOREFINE_LORA_BLOCK_SCALINGS must decode to a list")
    scalings = [float(value) for value in values]
  else:
    scaling = _env_float("MIMICX_AUTOREFINE_LORA_SCALING", 1.0)
    scalings = [scaling for _ in range(max_rank)]
  if len(scalings) < max_rank:
    scalings.extend([scalings[-1] for _ in range(max_rank - len(scalings))])
  return torch.tensor(scalings[:max_rank], dtype=torch.float32)


class RankOneLoraBlock(nn.Module):
  """Single rank-1 LoRA increment for the policy action head."""

  def __init__(self, input_dim: int, output_dim: int) -> None:
    super().__init__()
    self.down = nn.Linear(input_dim, 1, bias=False)
    self.up = nn.Linear(1, output_dim, bias=False)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(self.down.in_features))
    nn.init.zeros_(self.up.weight)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    return self.up(self.down(features))


class AutoRefineLoraMLPModel(MLPModel):
  """Frozen base actor plus per-reference rank-1 LoRA increments.

  The base G1 actor, observation normalizer, and action distribution parameters
  are frozen. Training modes are selected with environment variables:

  - ``MIMICX_AUTOREFINE_LORA_MODE=block`` trains exactly one rank-1 block.
  - ``MIMICX_AUTOREFINE_LORA_MODE=gate`` freezes all LoRA blocks and trains one
    scalar observation-conditioned gate over the complete active LoRA branch.
  - ``MIMICX_AUTOREFINE_LORA_MODE=frozen`` enables inference-only frozen LoRA.
  """

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.reference_id = os.environ.get("MIMICX_AUTOREFINE_LORA_REFERENCE_ID", "")
    self.lora_mode = os.environ.get("MIMICX_AUTOREFINE_LORA_MODE", "block").lower()
    self.max_rank = _env_int("MIMICX_AUTOREFINE_LORA_MAX_RANK", 8)
    if self.max_rank < 1:
      raise ValueError("MIMICX_AUTOREFINE_LORA_MAX_RANK must be >= 1")
    self.active_rank = max(
      0,
      min(
        self.max_rank,
        _env_int("MIMICX_AUTOREFINE_LORA_ACTIVE_RANK", 1),
      ),
    )
    default_train_block = max(0, self.active_rank - 1)
    self.train_block = _env_int(
      "MIMICX_AUTOREFINE_LORA_TRAIN_BLOCK", default_train_block
    )
    self.reinit_train_block = _env_bool(
      "MIMICX_AUTOREFINE_LORA_REINIT_TRAIN_BLOCK", False
    )
    self.gate_enabled = self.lora_mode == "gate" or _env_bool(
      "MIMICX_AUTOREFINE_LORA_GATE_ENABLED", False
    )

    final_index, final_layer = self._find_final_linear_layer()
    self._lora_final_index = final_index
    self._lora_input_dim = final_layer.in_features
    self._lora_output_dim = final_layer.out_features
    self.lora_blocks = nn.ModuleList(
      RankOneLoraBlock(self._lora_input_dim, self._lora_output_dim)
      for _ in range(self.max_rank)
    )
    self.scalar_gate = nn.Linear(self.obs_dim, 1)
    nn.init.zeros_(self.scalar_gate.weight)
    nn.init.constant_(
      self.scalar_gate.bias,
      _env_float("MIMICX_AUTOREFINE_LORA_GATE_INIT_BIAS", -4.0),
    )
    self.register_buffer("lora_block_scalings", _parse_scalings(self.max_rank))
    self.configure_autorefine_lora()

  def _find_final_linear_layer(self) -> tuple[int, nn.Linear]:
    for index in range(len(self.mlp) - 1, -1, -1):
      layer = self.mlp[index]
      if isinstance(layer, nn.Linear):
        return index, layer
    raise TypeError("AutoRefineLoraMLPModel requires an MLP with a Linear output layer")

  def configure_autorefine_lora(self) -> None:
    """Apply the freeze/train mask for the current AutoRefine stage."""
    for param in chain(
      self.mlp.parameters(),
      self.obs_normalizer.parameters(),
      self.distribution.parameters() if self.distribution is not None else (),
      self.scalar_gate.parameters(),
    ):
      param.requires_grad_(False)
    for block in self.lora_blocks:
      for param in block.parameters():
        param.requires_grad_(False)

    if self.lora_mode == "block":
      if not 0 <= self.train_block < self.active_rank:
        raise ValueError(
          "MIMICX_AUTOREFINE_LORA_TRAIN_BLOCK must be inside active rank "
          f"[0, {self.active_rank})"
        )
      for param in self.lora_blocks[self.train_block].parameters():
        param.requires_grad_(True)
      self.gate_enabled = False
    elif self.lora_mode == "gate":
      for param in self.scalar_gate.parameters():
        param.requires_grad_(True)
      self.gate_enabled = True
    elif self.lora_mode in {"frozen", "eval", "inference"}:
      pass
    else:
      raise ValueError(
        "MIMICX_AUTOREFINE_LORA_MODE must be one of block, gate, frozen"
      )

  def _base_and_lora(
    self, latent: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    features = latent
    base_output = None
    lora_features = None
    for index, layer in enumerate(self.mlp):
      if index == self._lora_final_index:
        lora_features = features
        base_output = layer(features)
      else:
        features = layer(features)
    if base_output is None or lora_features is None:
      raise RuntimeError("Failed to capture LoRA insertion point")

    lora_output = torch.zeros_like(base_output)
    for block_index in range(self.active_rank):
      lora_output = lora_output + self.lora_block_scalings[block_index] * self.lora_blocks[
        block_index
      ](lora_features)
    return base_output, lora_output

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
    latent = self.get_latent(obs, masks, hidden_state)
    base_output, lora_output = self._base_and_lora(latent)
    if self.gate_enabled:
      gate = torch.sigmoid(self.scalar_gate(latent))
      mlp_output = base_output + gate * lora_output
    else:
      mlp_output = base_output + lora_output
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output

  def update_normalization(self, obs: TensorDict) -> None:
    """Keep the native actor normalizer frozen during AutoRefine."""
    return None

  def get_extra_state(self) -> dict[str, Any]:
    return {
      "reference_id": self.reference_id,
      "lora_mode": self.lora_mode,
      "max_rank": self.max_rank,
      "active_rank": self.active_rank,
      "train_block": self.train_block,
      "gate_enabled": self.gate_enabled,
    }

  def set_extra_state(self, state: dict[str, Any]) -> None:
    self.reference_id = str(state.get("reference_id", self.reference_id))

  def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
    incompatible = super().load_state_dict(state_dict, strict=False, assign=assign)
    allowed_prefixes = ("lora_blocks.", "scalar_gate.", "lora_block_scalings")
    missing = [
      key
      for key in incompatible.missing_keys
      if key != "_extra_state" and not key.startswith(allowed_prefixes)
    ]
    unexpected = [
      key
      for key in incompatible.unexpected_keys
      if key != "_extra_state" and not key.startswith(allowed_prefixes)
    ]
    if strict and (missing or unexpected):
      raise RuntimeError(
        "Error(s) in loading AutoRefineLoraMLPModel state_dict: "
        f"missing={missing}, unexpected={unexpected}"
      )
    self.reinitialize_train_block_if_requested()
    self.configure_autorefine_lora()
    return incompatible

  def reinitialize_train_block_if_requested(self) -> None:
    """Freshly initialize the currently trained LoRA block after checkpoint load."""
    if not self.reinit_train_block or self.lora_mode != "block":
      return
    if not 0 <= self.train_block < self.max_rank:
      raise ValueError(
        "MIMICX_AUTOREFINE_LORA_TRAIN_BLOCK must be inside max rank "
        f"[0, {self.max_rank}) before reinitialization"
      )
    self.lora_blocks[self.train_block].reset_parameters()

  def as_jit(self) -> nn.Module:
    return _TorchAutoRefineLoraMLPModel(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxAutoRefineLoraMLPModel(self, verbose)

  def autorefine_lora_metadata(self) -> dict[str, Any]:
    trainable_names = [
      name for name, param in self.named_parameters() if param.requires_grad
    ]
    return {
      "reference_id": self.reference_id,
      "mode": self.lora_mode,
      "max_rank": self.max_rank,
      "active_rank": self.active_rank,
      "train_block": self.train_block if self.lora_mode == "block" else None,
      "gate_enabled": self.gate_enabled,
      "reinit_train_block": self.reinit_train_block,
      "block_scalings": [float(value) for value in self.lora_block_scalings.tolist()],
      "trainable_actor_parameters": trainable_names,
    }


class _ExportAutoRefineLoraBase(nn.Module):
  def __init__(self, model: AutoRefineLoraMLPModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.mlp = copy.deepcopy(model.mlp)
    self.lora_blocks = copy.deepcopy(model.lora_blocks)
    self.scalar_gate = copy.deepcopy(model.scalar_gate)
    self.register_buffer("lora_block_scalings", model.lora_block_scalings.detach().cpu())
    self.active_rank = int(model.active_rank)
    self.gate_enabled = bool(model.gate_enabled)
    self.final_index = int(model._lora_final_index)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.input_size = model.obs_dim

  def _forward_mlp(self, latent: torch.Tensor) -> torch.Tensor:
    features = latent
    base_output = None
    lora_features = None
    for index, layer in enumerate(self.mlp):
      if index == self.final_index:
        lora_features = features
        base_output = layer(features)
      else:
        features = layer(features)
    if base_output is None or lora_features is None:
      raise RuntimeError("Failed to capture LoRA insertion point")
    lora_output = torch.zeros_like(base_output)
    for block_index in range(self.active_rank):
      lora_output = lora_output + self.lora_block_scalings[block_index] * self.lora_blocks[
        block_index
      ](lora_features)
    if self.gate_enabled:
      lora_output = torch.sigmoid(self.scalar_gate(latent)) * lora_output
    return self.deterministic_output(base_output + lora_output)


class _TorchAutoRefineLoraMLPModel(_ExportAutoRefineLoraBase):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward_mlp(self.obs_normalizer(x))

  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxAutoRefineLoraMLPModel(_ExportAutoRefineLoraBase):
  is_recurrent: bool = False

  def __init__(self, model: AutoRefineLoraMLPModel, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._forward_mlp(self.obs_normalizer(x))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


class AutoRefineLoraPPO(PPO):
  """PPO variant that defaults to base-checkpoint actor/critic loading only."""

  def __init__(self, *args, optimizer: str = "adam", learning_rate: float = 0.001, **kwargs):
    super().__init__(*args, optimizer=optimizer, learning_rate=learning_rate, **kwargs)
    self._optimizer_name = optimizer
    trainable = [
      param
      for param in chain(self.actor.parameters(), self.critic.parameters())
      if param.requires_grad
    ]
    if not trainable:
      raise RuntimeError("AutoRefineLoraPPO has no trainable parameters")
    self.optimizer = resolve_optimizer(optimizer)(trainable, lr=learning_rate)

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    if load_cfg is None:
      load_cfg = {
        "actor": True,
        "critic": _env_bool("MIMICX_AUTOREFINE_LORA_LOAD_CRITIC", True),
        "optimizer": _env_bool("MIMICX_AUTOREFINE_LORA_LOAD_OPTIMIZER", False),
        "iteration": _env_bool("MIMICX_AUTOREFINE_LORA_LOAD_ITERATION", False),
        "rnd": False,
      }
    load_iteration = super().load(loaded_dict, load_cfg, strict)
    if hasattr(self.actor, "configure_autorefine_lora"):
      self.actor.configure_autorefine_lora()
    return load_iteration


class AutoRefineLoraMotionTrackingOnPolicyRunner(MotionTrackingOnPolicyRunner):
  """Motion-tracking runner that records AutoRefine LoRA stage metadata."""

  def save(self, path: str, infos=None):
    actor = self.alg.get_policy()
    lora_infos = {}
    if hasattr(actor, "autorefine_lora_metadata"):
      lora_infos["autorefine_lora"] = actor.autorefine_lora_metadata()
    super().save(path, {**(infos or {}), **lora_infos})
