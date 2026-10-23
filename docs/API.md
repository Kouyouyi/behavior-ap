# 外部控制器接口

`behavior_ap_sim.RadioEnvironment` 只管理单个、固定注册的收音机环境。每个 episode 在新进程中创建，所有环境调用使用启动它的线程。Planner、网络请求、动作预测及策略记忆由外部程序实现。

## 创建与推进

```python
env = RadioEnvironment(output, variant="ood", max_steps=10000,
                       render_profile="fast", headless=True, write_video=True)
env.start()
obs = env.capture(new_observation_directory)
advanced = env.step(action)
```

- `variant` 只接受 `original` / `ood`，不能覆盖任务、实例、种子、摆放或机器人配置。
- `capture()` 返回 `state`（官方61维原始proprio）、`image_paths`（按head、left_wrist、right_wrist顺序）及 `step`、`captured_ns`。图像为RGB，分辨率分别448×448、224×224、224×224。每次保存使用新目录。
- `step()` 接受一个有限的23维数组，转换成float32。顺序是base[0:3]、trunk[3:7]、arm_left[7:14]、gripper_left[14]、arm_right[15:22]、gripper_right[22]。动作单位、归一化和关节控制方式以包内 `configs/r1pro.yaml` 为准，不把它解释成任意末端笛卡尔动作。
- 返回 `True` 表示实际推进了一个30Hz任务步；停止后返回 `False`。异常后不得自动重试同一个物理动作，因为异常可能发生在物理已推进、录像尚未写完时。
- `steps` 是当前进程已执行的任务步数，初始化摆放/朝向稳定步不计入。`done`、`success`、`termination_reason` 分开提供结束状态、官方成功和结束原因。
- `capture()` 中的 `scorer_private.json` 是单独落盘的评测证据，不在返回给控制器的观测内。不要把目标世界坐标、分割掩码或私有评测文件传给待评策略。

API直接调用前应设置官方 `OMNIGIBSON_DATA_PATH` 环境变量；`BEHAVIOR_ASSET_ROOT` 别名仅由CLI启动器转换。

`close(reason)` 刷新并验证录像，写 `result.json`。无动作初始化不产生空视频；启用 `write_video=True` 时，有动作必须有完整录像。`shutdown()` 关闭Isaac，它可能原生退出整个进程。调用者应先保存所有结果，再调用它。推荐使用README中的双层 `finally`。

## GPU与外部推理

CLI自动持有GPU锁；直接用API时可 `with behavior_ap_sim.resources.gpu_lease(): ...`。多个本地项目必须显式设置相同的 `BEHAVIOR_GPU_LOCK_PATH` 才能共享锁。等待模型响应期间不调用 `step()`，不会推进任务物理。不要从HTTP处理线程直接调用环境。

`fast` 模式在每个新观测步刷新传感器，不推进物理。调用者自行决定观测频率；动作文件CLI的 `--capture-every` 默认8。重现同一轨迹时应固定动作、初始状态与观测频率。

## 可见性审计与状态交接

`initial_visibility_audit=True` 用于单独的初始场景检查：三相机必须都没有目标像素，才接受OOD初始不可见。该版本Isaac挂载分割图后不能安全恢复物理推进，因此此模式禁止 `step()` 和 `save_handoff()`。

`verify_approach(new_directory)` 是终端审计：实际渲染目标至少16像素，机器人底座到目标水平距离不超过1米。它只判断几何与可见性，不替外部Planner声明reached，也不意味着官方开机任务成功。调用后禁止继续物理；必须在此前导出需要交接的状态。

`save_handoff(new_directory)` 保存完整仿真状态、对象顺序、场景身份、SHA256及机器人/目标指纹。**只允许尚未结束且官方任务未成功的阶段交接**：源环境的 `done` 和 `success` 必须明确为 `False`，元数据写入 `source_task_done=false`、`source_official_success=false`。官方 `BehaviorTask.success` 缓存不包含在序列化的仿真状态中，因此已经结束或成功的episode不能通过此接口恢复。

另一新进程先创建相同variant并 `start()`，再 `restore_handoff(directory)`。目标环境同样必须未结束且官方未成功；缺少上述两个源状态标记的旧交接文件会被拒绝。恢复只允许在0任务步时尝试一次，检查完整状态 `atol=1e-4, rtol=1e-5`、指纹和注册身份，恢复过程中不推进物理。校验失败后禁止继续动作或导出。任务步预算由外部调度器减去源阶段的 `source_steps`，本包不会自动复用策略历史或动作。

## 命令行证据

`behavior-sim` 保存输入动作SHA、`actions.npy`、逐步submitted/ACK、按间隔观测、`execution_summary.json`、`sim/result.json`与两路H264视频。自动重放始终关闭。原生进程退出码为0但缺少结束证据，也判为基础设施错误。
