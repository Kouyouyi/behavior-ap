# behavior-ap：独立BEHAVIOR仿真环境

本仓库只提供仿真环境，不包含 Harness、Planner、VLA、模型权重或推理服务。任何外部控制器都可以通过 Python API 获取相机和机器人状态，再提交23维动作。无模型时也能打开场景、初始化或运行外部动作文件。

当前接入本机官方 **2026 BEHAVIOR-1K Challenge的全部100个任务**，每个任务均提供原始ID场景加载入口；OOD只注册经确认调好的收音机场景。

| 范围 | 数量 | 说明 |
|---|---:|---|
| 官方ID任务 | 100 | 官方任务编号0–99，每个任务都有 `original` / `id` 入口 |
| 官方房屋/场所布局 | 7 | 多个任务复用同一房屋布局；不把100任务说成100套独立房屋 |
| 原始任务实例 | 32,000 | 每任务300个train + 20个public_test；保留官方摆放和机器人起点 |
| 已注册OOD | 1 | 仅收音机 `radio_on_tv_cabinet_facing_room`，public_test/301、seed20261020、初始朝向偏移180° |

默认打开各任务 `public_test` 的实际实例ID **301**。`public_test`可选301–320；`train`中任务0–49为0–299，任务50–99为1–300。这里的 `--instance-id` 是资产中的实际编号，不是官方评测脚本0–19的测试集索引。

完整任务列表见 [ID_TASKS.md](docs/ID_TASKS.md)，官方索引及来源指纹见 `behavior_ap_sim/configs/official_tasks.json`。本机挑战实例库只覆盖上述100任务，不能据此宣称BEHAVIOR-1K所有1000项活动都有可运行资产。未提供的hidden_test/val不注册。未来OOD需用户确认后显式加入 `task_registry.json`；不会导入旧批测中未调好的OOD。

官方BDDL成功、搜索接近和场景初始化分别记录。注册原场景表示提供官方加载入口，不表示所有任务已完成机器人策略评测。

## 环境安装

先按 [BEHAVIOR-1K 官方说明](https://github.com/StanfordVL/BEHAVIOR-1K) 安装并授权资产。本机验证环境为 Python 3.11、OmniGibson 3.9.3、Isaac Sim 5.1.0、PyTorch 2.7.0+cu128、RTX 3090；版本及源码提交见 `behavior_ap_sim/configs/simulator-source.json`。需要官方环境提供 `torch`、`cv2`、`numpy`、`scipy`、`omegaconf`，以及视频编码包 `av` 和系统 `ffmpeg`/`ffprobe`。

```bash
git clone git@github.com:Kouyouyi/behavior-ap.git
cd behavior-ap
conda activate behavior-2026  # 或自己的已安装 OmniGibson/Isaac 的环境
python -m pip install --no-deps -e .
export BEHAVIOR_ASSET_ROOT=/path/to/BEHAVIOR-1K
export OMNIGIBSON_DATA_PATH="$BEHAVIOR_ASSET_ROOT"  # 直接使用 Python API 时也需要
```

资产根目录应含 `2026-challenge-task-instances`、`behavior-1k-assets`、`omnigibson-robot-assets`。本仓库不包含、不会下载或修改源资产。新机器必须自行安装官方环境，`pip install` 本包不会安装一整套 Isaac。

本机可直接复用 `/home/cody/miniconda3/envs/behavior-2026/bin/python`。资产若仅root可读，启动器使用本机已有受控权限helper或标准`sudo`，不会保存任何密码；资产可由普通用户读取时不提权。无需克隆旧 Harness 仓库，不需要其 `deps` 软链。

## 打开原生交互窗口

```bash
./scene --list                           # 数量汇总和100任务列表
./scene picking_up_trash                 # 默认 original / public_test / 301
./scene 1 original --instance-id 302      # 同任务另一个官方实例
./scene 50 id --split train --instance-id 1  # freeze_fruit训练实例
./scene radio original                   # 保留原收音机命令
./scene radio ood                        # 仅这个任务有已注册OOD
```

窗口初始暂停，可鼠标观察、选中物体并用移动/旋转工具调整。修改停留在USD session layer，不写回官方资产。此窗口用于检查场景，手工修改不属于有效评测。

安装包后也可用 `behavior-scene picking_up_trash original` 或 `behavior-scene radio ood`。命令会等待其他GPU计算进程结束，并要求至少20000MiB空闲显存；`--no-wait`在忙碌时返回75。

## 无模型初始化与外部动作

```bash
# 默认只初始化、保存初始观测并退出，不调用任何模型
./simulate picking_up_trash original --output outputs/id_init_01
./simulate 50 id --split train --instance-id 1 --output outputs/train_init_01

# 独立初始可见性审计；启用分割图后不可继续物理动作
./simulate radio ood --initialize-only --initial-visibility-audit \
  --output outputs/ood_visibility_01

# 执行其他程序生成的有限数值 N×23 float32 动作文件
./simulate radio ood --actions /path/to/actions.npy \
  --render-profile fast --output outputs/ood_actions_01
```

安装包后的命令为 `behavior-sim`。`--dry-run`只显示配置，不加载Isaac。`--capture-every`默认8，控制外部动作执行期间保存观测的间隔。每个运行必须使用新输出目录；执行前校验动作格式与SHA，保存原始动作、逐步提交/ACK、初末观测、官方结果和两路H264录像。基础设施或编码失败单列，退出检查不依赖Isaac的进程退出码，不自动重放动作。提前触发官方成功或环境结束时停止，实际动作数记录在 `execution_summary.json`。

## 外部控制器接入

```python
from behavior_ap_sim import BehaviorEnvironment

env = BehaviorEnvironment("outputs/external_01/sim", task="picking_up_trash",
                       variant="original", split="public_test", instance_id=301,
                       max_steps=10000, render_profile="fast")
try:
    env.start()
    observation = env.capture("outputs/external_01/observation_00000")
    # observation: 61维原始proprio、head/left_wrist/right_wrist图像路径及step
    # action = your_controller(observation)  # 在本仓库外运行的控制器
    # env.step(action)  # 只推进一次30Hz任务控制步
finally:
    try:
        env.close("external_controller_finished")
    finally:
        env.shutdown()
```

`RadioEnvironment`兼容旧接口（默认radio/ood）；新控制器使用通用 `BehaviorEnvironment`。普通ID任务不假定只有一个搜索目标，可见性/接近审计只开放给有显式目标注册的场景。

所有环境调用必须在创建/启动环境的同一线程进行；模型或网络等待期间不调用 `step`，物理就不会主动推进。23维动作是原R1Pro controller输入：base 0:3、trunk 3:7、左臂7:14、左夹爪14、右臂15:22、右夹爪22。夹爪仍采用原配置的assisted grasp，不宣称改成了纯接触抓取。

Python API不自动占用GPU锁；调用者应使用 `behavior_ap_sim.resources.gpu_lease()` 或自行安排独占GPU。CLI自动执行此检查。每个环境在新进程中创建；关闭Isaac可能原生退出整个进程，因此外部控制器必须在 `shutdown()` 前保存自己的结果。交接状态、可见性等评测辅助接口见 [API说明](docs/API.md)。

## 渲染配置

| `render_profile` | 行为 |
|---|---|
| `baseline`（默认） | 原渲染与同步H264录像 |
| `async_video` | 原渲染，有界队列异步编码，队列满时背压、不丢帧 |
| `fast` | 异步编码、固定第三人称相机，移除每动作额外渲染，在观测采集边界显式刷新 |

不改控制频率、物理求解器、碰撞或官方成功判定。`fast`改变第三人称取景和录像时序，不能假定像素等价；完整录像每个任务动作保留一帧，后续观测刷新不会回填已编码视频。

设计参考 [Isaac-sim-renderer 的固定提交](https://github.com/rascal-yang/Isaac-sim-renderer/tree/eccdecde583b2e4fc74ef8bb2cc84ba9d6dc6d02)。没有引入其G1 serializer，也没有把在线视觉闭环改成离线轨迹渲染。拆分前480步实测执行部分73.71秒→46.00秒、约1.60倍，含初始化约1.16倍；这是旧整合环境的测量，独立包验证另见 [验证记录](docs/VALIDATION.md)，不能当作新的开机成功率。

## 配置和测试

| 环境变量 | 用途 |
|---|---|
| `BEHAVIOR_PYTHON` | 已安装官方仿真依赖的Python解释器 |
| `BEHAVIOR_ASSET_ROOT` / `OMNIGIBSON_DATA_PATH` | 官方资产根目录 |
| `BEHAVIOR_OUTPUT_ROOT` | 默认输出目录 |
| `BEHAVIOR_GPU_LOCK_PATH` | 默认 `/tmp/behavior-ap-gpu.lock`；多个项目协同时显式使用同一个锁 |
| `BEHAVIOR_NO_SUDO=1` | 禁止自动提权 |
| `DISPLAY`、`XAUTHORITY` | 桌面显示与认证配置 |

```bash
python -m unittest discover -s tests -v
# 全量静态检查官方32000实例与模板，不启动Isaac
python scripts/catalog_official_assets.py --asset-root "$BEHAVIOR_ASSET_ROOT" \
  --report outputs/official_asset_check.json
```

测试含依赖隔离、任务白名单、动作ACK、交接身份、GPU锁和录像收尾故障传播；轻量测试不启动Isaac。模型、数据、资产、视频与本地凭据均不纳入Git。

当前ID扩展验证范围见 [ID验证记录](docs/ID_VALIDATION.md)。官方基础模板按原实现加载，所选split决定后续TRO状态；没有重新采样或人工移动ID任务物体。
