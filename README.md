# behavior-ap：独立收音机仿真环境

本仓库只提供仿真环境，不包含 Harness、Planner、VLA、模型权重或推理服务。任何外部控制器都可以通过 Python API 获取相机和机器人状态，再提交23维动作。无模型时也能打开场景、初始化或运行外部动作文件。

当前只注册已调好的 `turning_on_radio` 任务：

| 名称 | 实例 | 种子 | 初始朝向偏移 | 场景 |
|---|---:|---:|---:|---|
| `radio original` | 301 | 20261020 | 0° | 官方原场景 |
| `radio ood` | 301 | 20261020 | 180° | 收音机在电视矮柜上、开关面朝房间，机器人初始背向目标 |

注册表位于 `behavior_ap_sim/configs/task_registry.json`。未加入柜内遮挡、其他任务或任意场景覆盖。成功仍由官方 `BehaviorTask.success` / BDDL 判断；看到目标、走近目标与打开收音机是不同结果。

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
./scene --list
./scene radio original
./scene radio ood
```

窗口初始暂停，可鼠标观察、选中物体并用移动/旋转工具调整。修改停留在USD session layer，不写回官方资产。此窗口用于检查场景，手工修改不属于有效评测。

安装包后也可用 `behavior-scene radio ood`。命令会等待其他GPU计算进程结束，并要求至少20000MiB空闲显存；`--no-wait`在忙碌时返回75。

## 无模型初始化与外部动作

```bash
# 默认只初始化、保存初始观测并退出，不调用任何模型
./simulate radio ood --output outputs/ood_init_01

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
from behavior_ap_sim import RadioEnvironment

env = RadioEnvironment("outputs/external_01/sim", variant="ood",
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
```

测试含依赖隔离、任务白名单、动作ACK、交接身份、GPU锁和录像收尾故障传播；轻量测试不启动Isaac。模型、数据、资产、视频与本地凭据均不纳入Git。
