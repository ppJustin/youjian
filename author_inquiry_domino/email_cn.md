主题：咨询 VLAct DOMINO 100k 微调复现的结果差距及数据末帧处理

作者您好：

我已经完成一版 VLAct QwenOFT DOMINO 微调（本地称为 v2），正常训练到 100,000 步，并使用该 100k 权重进行了 clean-dynamic 任务测试。希望请您帮助确认与官方训练及评测设置的差异。

训练日志明确记录 Training complete 和 100000/100000。评测 checkpoint 位于 vlact_domino_qwen3oft_angularfix_v2_wrap，附带配置的 run_id 仍为 vlact_domino_qwen3oft_angularfix（两处配置相同；v2 目录是评测使用的目录名）。

当前找到 35 个任务的评测日志：33 个任务达到 100 episodes，place_can_basket 为 97，put_object_cabinet 为 77。合计 180/3474 次成功，合并 SR 约 5.18%；按各任务日志末尾的一位小数 MS 计算，未加权均值约 24.47。这些是现有日志的暂时汇总，并非完全完成的标准 35×100 评测。与官方模型卡报告 SR 18.50 / MS 34.20 相比存在明显差距。

我比对了 v2 实际保存的 config.yaml 和官方发布模型的 training_config.original.yaml：

1. VLM 学习率 1e-5，动作头/base 学习率 1e-4，warmup 5000，cosine_with_min_lr（min_lr 5e-7），AdamW betas [0.9, 0.95]、eps 1e-8、weight_decay 1e-8，以及 100k 步、每卡 batch 16、动作未来窗口31、seed42、随机初始化动作头、加载预训练100k和reset_steps，均一致。
2. v2 每10000步保存，官方发布配置每20000步保存。v2 额外显式启用 gradient checkpointing；官方发布配置未记录该项（未记录不一定代表关闭）。
3. v2 使用本地 Qwen3-VL-4B-Instruct-Action。官方原始配置记录 StarVLA/Qwen3-VL-4B-Instruct，而便携 config.yaml 记录 Action 基座，请问两者是否等价？预训练权重是否会覆盖所有相关 VLM 参数？
4. 最重要的数据差异：v2 使用 domino_clean_wrap，5250条轨迹、955478个transitions；官方统计是5250条轨迹、950228个transitions，正好多5250个，即每条轨迹一帧。v2 没有去末帧，也没有显式指定官方 pretrained_stats_path，而是使用本地生成的统计。请问官方是否在转换时保留 N-1 个观测/动作对？该额外末行会怎样影响训练样本、归一化统计、末尾32步动作窗口及padding？
5. v2 训练时本地添加了 robotwin* 角度标签映射，但当前代码优先读取 robot_tag=new_embodiment，官方已有该标签，因此按此路径新增映射应不改变损失。请确认官方 DOMINO 是否使用 new_embodiment 和 shortest_angular_joint_loss=True。
6. 双方保存配置都是 freeze_modules 的布尔值 true，但仓库冻结函数仅处理字符串模块名。请问官方训练是否实际上全参数训练？
7. 评测 launcher 根据模型目录名是否含 wrap 选择 wrap/plain 变体；v2_wrap 使用 wrap。请确认官方发布权重要求的动作后处理、动作排列、角度wrap、chunk执行长度、图像尺寸/视角、历史输入、动态系数、种子及 expert_check 设置。是否建议先用官方发布权重在同一环境做对照，以区分训练和评测问题？

当前另准备了一个新脚本（附件 train_domino_qwen3oft_official_local.sh），直接读取官方原始配置，保留本地去末帧并指定官方统计。它不是上述 v2 的实际训练脚本，请问这样的数据适配是否正确，还是应该重新转换数据？

附件包含 v2 实际配置、原始启动脚本、训练完成证据、逐任务评测汇总、评测脚本、双方统计和官方配置。希望您指出最可能导致效果差距的因素，并提供官方数据转换/校验及评测设置建议。

谢谢！
