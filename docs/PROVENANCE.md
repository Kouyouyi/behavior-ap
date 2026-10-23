# 场景注册来源

本独立项目于2026-10-23从已调试的收音机环境抽取。旧项目的Planner、模型客户端、批测脚本、模型权重、生成视频与源资产均没有纳入本仓库。运行时仅需本包和单独安装的官方仿真依赖。

注册范围是官方 `turning_on_radio`（task0）、public_test实例301、种子20261020。原场景不加摆放修改；OOD场景使用 `radio_on_tv_cabinet_facing_room`，将收音机放在电视矮柜上、开关朝房间，并将机器人起始朝向旋转180度。其他任务和柜内遮挡未注册。

OOD来源记录为 `champion_start_comparison_20261020_v3/final_comparison.json`。历史对照中，Harness搜索480步后，冠军权重继续5118步，官方BDDL成功，总5598步；直接冠军组10000步未成功。它支持此场景已有有效运行经验，只是一个场景和种子，不代表泛化成功率，也不是本独立包重新调用模型得到的结果。

参考环境的物理控制、官方成功定义、完整状态交接和录像实现被保留；独立包新验收见 [VALIDATION.md](VALIDATION.md)。注册表中的历史来源字段仅用于追溯，不参与策略输入和官方任务判定。

依赖源码提交见 `behavior_ap_sim/configs/simulator-source.json`。渲染优化参考 `rascal-yang/Isaac-sim-renderer` 的 `eccdecde583b2e4fc74ef8bb2cc84ba9d6dc6d02`，保留本环境的OmniGibson在线物理和原生相机，不使用参考项目的G1离线serializer。资产和外部组件许可见 [NOTICE.md](../NOTICE.md)。
