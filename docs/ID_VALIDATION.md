# 0.2.0官方ID任务接入验证：2026-10-23

本次将独立仿真入口扩展到已安装官方2026 Challenge的全部100个任务，保留原始摆放、机器人起点和BDDL成功条件。没有增加新OOD；只有此前确认的收音机OOD继续注册。结构化记录见 [id_validation_20261023.json](id_validation_20261023.json)。

## 全量静态验证

- task.jsonl、B100_task_misc.csv、available_tasks.yaml的100个任务编号、名称和房屋布局交叉一致。
- 7种布局、30,000个train实例、2,000个public_test实例；全部32,000个实例的入口解析、split与实际编号验证通过。
- 全量解析32,399个JSON：32,000个实例文件与399个模板；没有缺失必需文件、无效JSON或多余实例文件。
- 缺少的1个bringing_water可选partial_rooms缓存不属于官方evaluator的加载依赖。保持官方train完整基础模板加所选split TRO的加载方式，没有生成替代资产。
- 86项回归检查通过，覆盖注册表、CLI/窗口参数、实际载入身份核对、观测、交接约束、逐步ACK与录像收尾；其中H264写出与完整解码为真实编码检查，其他轻量检查不启动Isaac。

完整范围是本机已安装的100任务挑战实例库，不是BEHAVIOR-1K所有1000活动定义。未分发的hidden_test和不存在的独立val split未注册。

## 独立安装包

真实启动从`/tmp`运行，Python路径指向解压后的0.2.0 wheel及已有官方依赖，不加入旧Harness源码。安装包包含100任务目录和机器人配置。隔离导入、100任务注册和训练实例解析通过；导入注册表不加载Torch/Isaac/Planner。

验证wheel：`behavior_ap-0.2.0-py3-none-any.whl`，SHA256：

```text
1d37576bfb2439282b7f10316974fccaa1b35091cfe79ec50e651d6530b8d32f
```

## 真实启动抽测

`freeze_fruit`（task50）、train实际实例1、`house_single_floor`：初始化与正常关闭通过，实际加载任务、房屋、定义ID和实例ID严格一致。保存61维proprio、448×448头部RGB、两路224×224腕部RGB和完整仿真状态。0任务动作，无模型推理；官方成功值为false，结束原因为`initialize_only`。本次验证了状态导出，没有执行恢复后的策略。

`installing_a_fax_machine`（task89）、public_test实际实例301、`office_cubicles_right`：原生Isaac窗口创建通过，实际载入身份验证通过，初始暂停、USD session layer编辑启用，ready后运行3秒并正常关闭；0任务动作、没有模型推理、无收尾错误。没有进行人工鼠标拖动验收。

本机证据目录：`outputs/id_extension/freeze_fruit_train1_v1`、`outputs/id_extension/fax_public301_gui_v1`。静态资产报告在`outputs/id_extension/asset_validation.json`，不将官方资产或生成图像/状态文件纳入Git。

这些检查证明全量注册/资产一致性和抽测场景的启动能力，不表示100任务都已逐一完成物理评测，也不表示机器人完成了原任务。先前收音机原场景、OOD动作与渲染性能证据保存在 [历史验证记录](VALIDATION.md)。
