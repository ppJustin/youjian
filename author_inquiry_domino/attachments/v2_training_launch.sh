#!/usr/bin/env bash
# =====================================================================
# train_domino_qwen3oft_dlc.sh
# VLAct Qwen3 OFT × DOMINO 微调 —— DLC / 无 conda activate 环境专用
#
# 修复内容: QwenOFT.py 的 AUTO_ANGULAR_JOINT_LOSS_GRIPPER_DIMS 已补全
#           robotwin_wrap32 等标签, shortest_angular_joint_loss 现在对
#           DOMINO 真正生效 (跨 ±π 关节不再被普通 L1 错误惩罚)。
#
# 复现目标: StarVLA/VLAct_Qwen3OFT_Domino_Finetune (SR 18.50 / MS 34.20)
# 起点:     VLAct_Qwen3_Pretrain steps_100000 (OFT head 随机初始化)
# 数据:     domino_clean_wrap (Clean_Dynamic 35 task, 32-step horizon)
# 步数:     100K, 每 10K 存一次
# =====================================================================
set -euo pipefail

# ─── 激活 conda 环境 ─────────────────────────────────────────────────
source /cpfs01/zijian.pan/miniforge3/etc/profile.d/conda.sh
conda activate starvla

# ─── 路径约定 ────────────────────────────────────────────────────────
export VLACT_ROOT=/cpfs01/zijian.pan/VLAct
export NCCL_SOCKET_IFNAME=eth0                                          # 本机/容器无 bond0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000                                                # 秒
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

cd "${VLACT_ROOT}"

# ─── 训练超参 (与官方 scripts/run_scripts/DOMINO/train_domino_qwen3oft.sh 一致) ───
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/DOMINO/train_files/starvla_train_domino.yaml
data_root_dir=./playground/Datasets/DOMINO
run_root_dir=./results/Checkpoints
data_mix=domino_clean_wrap
action_chunk_size=32
future_action_window_size=31
image_size_buckets='[[320,180],[280,210]]'
run_id=vlact_domino_qwen3oft_angularfix                                  # 新 run_id, 不覆盖旧权重
pretrained_ckpt=./results/Checkpoints/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt
num_processes=${NUM_PROCESSES:-8}
main_port=${MAIN_PROCESS_PORT:-29500}

# ─── 输出目录 + 日志 (写持久盘, 不写 /tmp) ───────────────────────────
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
# 脚本自拷贝 (便于复现)
cp -f "$0" "${output_dir}/" 2>/dev/null || true

# 日志文件: 带时间戳, 持久保存
LOG_FILE="${output_dir}/train_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] 训练日志: ${LOG_FILE}"
echo "[INFO] Python: $(which python)"
echo "[INFO] 输出目录: ${output_dir}"
echo "[INFO] 修复确认: AUTO_ANGULAR_JOINT_LOSS_GRIPPER_DIMS 已含 robotwin_wrap32"

# ─── 启动训练 ────────────────────────────────────────────────────────
exec accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  --main_process_port ${main_port} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.future_action_window_size ${future_action_window_size} \
  --framework.action_model.action_horizon ${action_chunk_size} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --datasets.vla_data.video_backend pyav \
  --datasets.vla_data.sequential_step_sampling False \
  --datasets.vla_data.image_size_buckets "${image_size_buckets}" \
  --trainer.shortest_angular_joint_loss True \
  --trainer.pretrained_checkpoint ${pretrained_ckpt} \
  --trainer.random_init_action_model True \
  --trainer.reset_steps True \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.learning_rate.base 1e-4 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_DOMINO \
  --wandb_entity "${WANDB_ENTITY:-your_name}" \
  2>&1 | tee "${LOG_FILE}"
