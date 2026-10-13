Subject: Questions about a completed VLAct DOMINO 100k reproduction and terminal-row handling

Dear VLAct authors,

I completed a QwenOFT DOMINO fine-tuning run (locally named v2) at 100,000 steps and evaluated its 100k checkpoint on clean-dynamic tasks. The training log confirms normal completion. The evaluation directory is vlact_domino_qwen3oft_angularfix_v2_wrap; its configuration retains the original training run_id vlact_domino_qwen3oft_angularfix.

I found logs for all 35 tasks: 33 reached 100 episodes, while place_can_basket reached 97 and put_object_cabinet 77. The currently logged totals are 180/3474 successes (pooled SR 5.18%). The unweighted mean of the rounded per-task MS values is approximately 24.47. This is a provisional log summary, not a completed 35×100 benchmark. Your model card reports SR 18.50 / MS 34.20.

Comparing the saved v2 configuration with the released training_config.original.yaml, the learning rates (VLM 1e-5, action/base 1e-4), 5000-step warmup, cosine scheduler with min_lr 5e-7, AdamW settings, batch size 16 per device, 100k target steps, seed 42, action windows, pretrained initialization, random action-head initialization, and reset_steps match.

Could you clarify these differences and questions?

1. Our checkpoint interval was 10k versus 20k in the released configuration. We explicitly enabled gradient checkpointing; the released configuration does not record that option.
2. Our base model is the local Qwen3-VL-4B-Instruct-Action. The original configuration lists Qwen3-VL-4B-Instruct, while the portable configuration lists the Action model. Are these equivalent for this initialization procedure?
3. Our dataset statistics contain 5250 trajectories and 955478 transitions, versus the official 5250/950228. This is exactly one extra row per trajectory. Our completed v2 run did not drop terminal rows and used locally generated statistics rather than an explicit official pretrained_stats_path. Does the official conversion produce N-1 observation/action pairs? How should the terminal row and the final 32-step action windows be handled?
4. We added robotwin* entries to the angular-loss mapping during this run. However, the current code prioritizes robot_tag=new_embodiment, which is already supported upstream, so these additions appear inactive for this data path. Is new_embodiment with shortest_angular_joint_loss=True the intended official setup?
5. Both saved configurations record boolean freeze_modules=true, while the current freezing function only handles string module paths. Was the official run full-parameter fine-tuning?
6. Our evaluation launcher selects wrap/plain according to whether the model directory name contains wrap. We used the wrap variant. Please confirm the expected action postprocessing, dimension ordering, chunk execution length, images/views, history inputs, dynamic coefficient, seeds, and expert_check settings. Would evaluating your released weights in the same environment be the recommended first diagnostic?

We also attach a revised script that reads the official original configuration and adds terminal-row filtering plus official statistics. This revised script was not used to train the completed v2 checkpoint. Is this adaptation appropriate, or should we reconvert the data?

Attachments include the actual v2 configuration and launch script, training completion evidence, per-task evaluation summary and launcher, both statistics files, and the official configuration. I would appreciate your advice on the most likely causes of the performance gap and the exact official conversion/evaluation settings.

Thank you for your help.
Best regards,
