#!/usr/bin/env bash
set -Eeuo pipefail

# Official VLAct QwenOFT DOMINO fine-tuning configuration.
# Local-only changes: filesystem paths and drop_domino_terminal_frame=True.

export VLACT_ROOT="${VLACT_ROOT:-/cpfs01/zijian.pan/VLAct}"
export STARVLA_PYTHON="${STARVLA_PYTHON:-/cpfs01/zijian.pan/conda-envs/starvla/bin/python}"
export STARVLA_ROOT="${VLACT_ROOT}"
export PATH="$(dirname "${STARVLA_PYTHON}"):${PATH}"
export PYTHONPATH="${VLACT_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

cd "${VLACT_ROOT}"

RUN_ID="vlact_domino_qwen3oft_official_local"
RUN_ROOT="${VLACT_ROOT}/results/Checkpoints"
OUT_DIR="${RUN_ROOT}/${RUN_ID}"
BASE_VLM="${VLACT_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action"
PRETRAIN="${RUN_ROOT}/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt"
OFFICIAL_STATS="${VLACT_ROOT}/playground/Pretrained_models/VLAct_Qwen3OFT_Domino_Finetune/dataset_statistics.json"
DATA_ROOT="${VLACT_ROOT}/playground/Datasets/DOMINO"
CONFIG="${VLACT_ROOT}/playground/Pretrained_models/VLAct_Qwen3OFT_Domino_Finetune/training_config.original.yaml"
DS_CONFIG="${VLACT_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml"

mkdir -p "${OUT_DIR}"
cp "$0" "${OUT_DIR}/"

"${STARVLA_PYTHON}" - <<'PY'
from pathlib import Path
import json, os, sys
assert sys.version_info[:2] == (3, 10), sys.version
root = Path(os.environ["VLACT_ROOT"])
required = [
    root / "playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/config.json",
    root / "results/Checkpoints/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt",
    root / "playground/Pretrained_models/VLAct_Qwen3OFT_Domino_Finetune/dataset_statistics.json",
    root / "playground/Datasets/DOMINO",
    root / "starVLA/config/deepseeds/deepspeed_zero2.yaml",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"Missing required inputs: {missing}")
stats = json.loads(required[2].read_text())["new_embodiment"]
print(f"official statistics: {stats['num_transitions']} transitions, {stats['num_trajectories']} trajectories")
PY

exec "${STARVLA_PYTHON}" -m accelerate.commands.launch \
  --config_file "${DS_CONFIG}" \
  --num_processes 8 \
  --main_process_port "${MAIN_PROCESS_PORT:-29501}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --is_debug False \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix domino_clean_wrap \
  --datasets.vla_data.drop_domino_terminal_frame True \
  --datasets.vla_data.pretrained_stats_path "${OFFICIAL_STATS}" \
  --trainer.pretrained_checkpoint "${PRETRAIN}" \
  --trainer.random_init_action_model True \
  --trainer.reset_steps True \
  --trainer.freeze_modules True \
  --run_root_dir "${RUN_ROOT}" \
  --run_id "${RUN_ID}" \
  --wandb_project starVLA_DOMINO \
  --wandb_entity "${WANDB_ENTITY:-your_name}" \
  2>&1 | tee "${OUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
