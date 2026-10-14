作者您好：

感谢您的建议。关于“先用最新版 starVLA 跑一下不用 VLAct pretrain 的结果，正常的话再换一下 VLM backbone”，我想确认是否理解正确：

1. 先使用最新版 starVLA，保留 Qwen3-VL backbone，不加载 `VLAct_Qwen3_Pretrain` 权重，随机初始化动作头，直接在 DOMINO 上训练和评测。
2. 如果结果正常，再换一个受支持的 VLM backbone，仍不加载 VLAct 预训练权重，比较训练效果。

也就是先确认不依赖 VLAct 预训练的训练流程是否正常，再判断 backbone 对效果的影响。请问是这个逻辑吗？另外，“正常”是否有可参考的成功率，以及第二步推荐使用哪个 backbone？

下面将已完成实验记为 v1，麻烦您看一看 config 是否和您的实验有出入，谢谢您。

```python
v1 = {
    "framework.name": "QwenOFT",
    "framework.qwenvl.base_vlm":
        "playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
    "trainer.pretrained_checkpoint":
        "./results/Checkpoints/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt",
    "trainer.random_init_action_model": True,
    "trainer.reset_steps": True,
    "trainer.is_resume": False,
    "trainer.freeze_modules": True,

    "datasets.vla_data.data_mix": "domino_clean_wrap",
    "datasets.vla_data.action_mode": "abs",
    "datasets.vla_data.per_device_batch_size": 16,
    "datasets.vla_data.image_size_buckets": [[320, 180], [280, 210]],
    "datasets.vla_data.sequential_step_sampling": False,
    "framework.action_model.action_dim": 14,
    "framework.action_model.future_action_window_size": 31,
    "framework.action_model.past_action_window_size": 0,

    "trainer.learning_rate.qwen_vl_interface": 1e-5,
    "trainer.learning_rate.action_model": 1e-4,
    "trainer.learning_rate.base": 1e-4,
    "trainer.gradient_accumulation_steps": 1,
    "trainer.max_train_steps": 100000,
    "trainer.num_warmup_steps": 5000,
    "trainer.lr_scheduler_type": "cosine_with_min_lr",
    "trainer.scheduler_specific_kwargs": {"min_lr": 5e-7},
    "trainer.shortest_angular_joint_loss": True,
    "trainer.enable_gradient_checkpointing": True,
    "seed": 42,
}
```

对于这个已完成的训练，我还有几个问题想向您请教：

1. 官方原始实验配置使用的是 `StarVLA/Qwen3-VL-4B-Instruct`，但是我目前只能下载到 `Qwen3-VL-4B-Instruct-Action`。请问这两个模型是否有区别，是否会影响复现官方训练结果？
2. 我从官方下载的数据集是 5250 条轨迹、955478 个 transitions，比官方统计多 5250 个，即每条轨迹多一帧。我目前的处理方式是每条轨迹减去最后一帧，并使用归一化统计。不知道这样是否会有影响？还是训练时无需减去这一帧，直接按照官方下载的数据集训练即可，不必对齐官方统计口径？
3. `freeze_modules` 保存为布尔值 `True`，我不清楚这个设置是否会实际冻结模块，想请您确认。

再次感谢您的指导！
