# 独立包验证：2026-10-23

实际验证使用本仓库生成的wheel，解压到独立目录，并从 `/tmp` 启动。运行时Python路径只指向该安装包和已有官方仿真依赖，不加入旧Harness源码。结构化记录见 [validation_20261023.json](validation_20261023.json)。

## 本次结果

| 检查 | 结果 |
|---|---|
| 轻量回归 | 64项通过，含真实H264编码与完整解码 |
| 打包 | wheel生成成功，任务JSON和R1Pro YAML包含在安装包中 |
| 导入隔离 | 导入API/注册表不加载Isaac、Torch、Planner或模型模块 |
| 原生Isaac窗口 | original窗口创建成功，初始暂停、USD session layer编辑启用，运行3秒后正常关闭；未做人工鼠标拖动验收 |
| original真实初始化 | 通过，0任务步，61维状态和三相机正常，完整交接状态导出成功 |
| OOD真实动作执行 | 新初始场景，480个外部23维动作全部执行，480条提交与480条ACK一一对应 |
| 拆分前后数值回归 | 每8步采样，共61次、每次61维proprio，最大绝对差0 |
| 两路录像 | robot_cameras 672×448、third_person 1280×720，各480帧/30fps，H264/yuv420p；ffmpeg完整解码通过 |

OOD运行使用 `fast`，含初始化的backend总墙钟时间170.90秒；480次action_step累计41.57秒，观测刷新累计2.79秒。计时分项不能直接相加成完整运行时间，也没有在本次独立包中重新测baseline，因此不据此宣称新的加速倍数。

本次只使用固定外部动作文件，**没有调用Planner或VLA，没有进行完整开机策略评测**。官方任务成功值为false，结束原因 `external_actions_complete`。机器人状态对齐不意味着所有场景对象状态或图像像素逐项相同。新的跨进程恢复路径有严格单测；本次真实运行验证了导出，没有重新执行恢复后的闭环策略。

本机证据保存在 `outputs/standalone_original_v1`、`outputs/standalone_ood_480_v1`、`outputs/standalone_gui_original_v1` 和 `outputs/standalone_validation.json`，不将视频、外部动作文件和官方资产推入Git。验证环境为RTX3090、Python3.11、OmniGibson3.9.3、Isaac5.1.0、PyTorch2.7.0+cu128；官方源码提交见包内 `configs/simulator-source.json`。

## 历史性能证据

拆分前整合环境的480步同动作对照：baseline执行73.71秒，fast执行46.00秒，约1.60倍；包含冷初始化约1.16倍。该历史数据仅说明加速实现的来源。场景历史策略结果见 [PROVENANCE.md](PROVENANCE.md)，与本次纯环境验证分别记录。
