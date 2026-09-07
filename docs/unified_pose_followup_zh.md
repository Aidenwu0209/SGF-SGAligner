# 三方案融合的第二轮优化与消融（2026-09-07）

本轮得到一个平均误差更低的**实验候选**，同时确认了上一轮收益损失的主要来源。相对 `af64e9f` 的新融合候选，本轮组合候选在两个可比较网格场景的平均 F-score 增加 **1.0345 个百分点**、Chamfer-L1 降低 **8.0602%**；三个有候选的场景合并计算平移 ATE 降低 **8.6457%**、旋转误差降低 **9.4170%**。但 `scene0046_01` 的 F-score 下降 **0.8709 个百分点**，所以没有全面提升。

三个候选仍未通过沿用的几何双改善门，第四个场景没有可接受的回环，**四个 committed 结果仍等于 DPV，最终交付结果的精度变化为 0**。新实现作为显式实验选项保存在 `develop`，没有替换默认配置或修改 `main`。本报告的数值和源码哈希见 [机器结果](unified_pose_followup_validation_20260907.json)。

## 本轮实际完成了什么

| 实验组 | 场景 / 臂数量 | 固定与变化 | 结果用途 |
|---|---:|---|---|
| factorial | 4 × 4 = 16 | 同一批历史 PnP 回环，只改变限权与全局缩放两个因素 | 定位融合后收益损失 |
| smooth_local | 4 × 1 = 4 | 同一批历史回环与限权，全局缩放改为局部平滑缩放 | 独立验证缩放机制 |
| fresh_refined | 4 × 1 = 4 | 重新运行 Hybrid36 / CLIP、双向 PnP、跨 family 共识与重融合；增加拒绝后的有界 ICP 恢复 | 检验回环覆盖与真实入口 |
| verified_local | 4 × 1 = 4 | 冻结上一组实际进入 PGO 的边、变换和权重，只切换局部平滑缩放 | 得到最终组合的受控后端结果 |

共 28 项场景/臂实验，其中 26 项产生候选并完整重融合。`fresh_refined` 的 `0030_02` 没有回环，完整重融合 DPV；`verified_local` 对该场景复用已封存的 DPV 点云并核验绑定，不伪造候选或再算一份“优化精度”。后者的候选 GT 数值保留 null。

同一场景的 DPV、RGB-D 像素、admitted frames、TSDF 参数及 GT 评价区域保持一致。完整输入为 11,882 帧，TSDF 使用所有帧，体素 2 cm、截断 8 cm、深度截断 4.5 m。每组先完成并封存所有推理结果，再由独立评价进程读取 GT；评价前后 552 项推理产物哈希不变。本轮四场景已经用于开发，不能作为未见场景泛化结论。

## 为什么单独方案收益大，组合后收益小

冻结旧回环的 2×2 实验给出了比“模块可能相互干扰”更具体的证据：

| 历史回环后端 | 三网格场景平均 F-score@5cm ↑ | 平均 Chamfer-L1 ↓ | 合并平移 ATE ↓ |
|---|---:|---:|---:|
| DPV 基线 | 34.0696% | 13.6353 cm | 0.7699 m |
| original_pnp：旧后端 | 48.4438% | 9.0801 cm | 0.1922 m |
| weight_only：只增加长跨度限权 | 48.8284% | 9.0693 cm | 0.1942 m |
| correction_only：只增加全局缩放 | 41.2387% | 11.1548 cm | 0.7110 m |
| combined_bounded：限权与全局缩放 | 41.2429% | 11.1514 cm | 0.7114 m |

在这批固定回环上，**大部分候选收益损失来自统一缩放**。一个局部的大修正会迫使整条轨迹乘同一个小系数，其他位置本来合理的漂移修正也被压小。`0000_00` 的全局系数只有 0.0625，其他三个场景为 0.5、0.25、0.25。

但不能直接取消约束：旧后端在 `0000_00` 的占用体素比只有 0.7188，限权单独使用也只有约 0.7198；全局缩放后约为 0.9898，确实缓解了过度收缩。旧高分也来自冻结的历史回环，未重新通过本轮更严格的配准验证，不能当作当前完整入口可直接部署的成绩。限权还可能改变 degree 选边排序；因此 `weight_only` 不只是对固定边残差乘一个系数。

## 新实现一：局部平滑缩放

`bounded.correction_scaling_policy: smooth_local` 为每个 anchor 求一个 0–1 的修正系数，并约束相邻系数作用后的修正平滑。在同样的 **0.25 m / 5° 绝对修正**及 **0.05 m / 2° 相邻修正**限制下，尽可能保留各位置可行的修正。

平移和旋转共用一个 anchor 系数。求解后仍传播到全部输入帧，重新做完整 SE(3)、帧身份、绝对修正与相邻修正审计；需要时执行原全局回溯。求解失败、审计失败或几何门失败仍回退 DPV。`selected_correction_scale` 在此模式下只是最后的全局回溯倍率，应结合 `selected_anchor_correction_scales` 看实际修正量。

冻结旧回环时，局部策略相对全局策略使合并平移/旋转误差降低 8.9831% / 9.4184%，平均 Chamfer 降低 8.2387%；平均 F-score 却从 41.2429% 降至 40.6681%（−0.5748 pp）。这说明轨迹更准和平均距离更低，并不保证 5 cm 阈值内的覆盖率同时提高。

## 新实现二：有界 ICP 恢复部分配准

新增 `registration.preconsensus_geometric_icp: true`。只有原始共识拒绝时，才对独立三维假设执行至多 0.10 m / 5° 的 ICP 细化，再尝试原共识。原本接受的路径保持原流程。

恢复时必须存在**未被 ICP 改写的 RGB-D PnP 证据**，获胜簇包含至少两个独立 family；ICP 不能充当新的独立证人。原 `2.5° / 0.05 m` 共识门、正反方向检查、支持门和后续累计修正上限继续生效。

四场景实际进入 PGO 的回环数从 **7 / 0 / 3 / 4** 变为 **8 / 0 / 4 / 4**。单独加入这个恢复步骤没有带来平均精度提升：`0046_00` 的 F-score 从 27.7089% 降到 27.3930%，`0046_01` 基本不变。回环数量本身不是精度指标。

`0030_02` 仍未恢复。两个历史关键回环在细化后与独立 PnP 仍有约 6.06–9.20 cm 平移差，超过 5 cm 共识门；其中一个方向最接近约 7.24 cm。这里的证据不足以支持为了保留旧边而放宽门限。

## 最终组合的精度对照

本表对照的是**上一轮完整融合候选 → 本轮恢复配准后、同边局部缩放候选**，不是 committed 结果。F-score、Precision、Recall 使用 5 cm 阈值，以下增减为百分点；距离与角度越小越好。

| 指标 | scene0046_00：上一轮 → 本轮 | 变化 | scene0046_01：上一轮 → 本轮 | 变化 |
|---|---:|---:|---:|---:|
| F-score@5cm | 27.7089% → 30.6488% | +2.9399 pp | 37.3115% → 36.4406% | −0.8709 pp |
| Precision@5cm | 24.2446% → 27.0787% | +2.8341 pp | 33.3557% → 32.8023% | −0.5534 pp |
| Recall@5cm | 32.3283% → 35.3033% | +2.9749 pp | 42.3318% → 40.9867% | −1.3451 pp |
| Chamfer-L1 | 15.1749 → 13.8034 cm | −9.0378% | 11.1796 → 10.4268 cm | −6.7333% |
| Chamfer-RMSE | 23.2245 → 22.0567 cm | −5.0285% | 19.1533 → 18.4709 cm | −3.5628% |
| 平移 ATE RMSE | 0.4273 → 0.3619 m | −15.2979% | 0.2753 → 0.2462 m | −10.5790% |
| 旋转 RMSE | 7.2357° → 6.6886° | −7.5608% | 9.7783° → 8.2087° | −16.0511% |

`scene0000_00` 平移 ATE 为 0.9955 → 0.9152 m（−8.0664%），旋转为 28.0467° → 25.4839°（−9.1376%）；该输入包没有 GT mesh，不提供 F-score。`scene0030_02` 没有候选，不混入候选聚合。

| 聚合指标 | DPV | 上一轮融合候选 | 本轮 ICP 恢复 + 全局缩放 | 本轮同边局部缩放 |
|---|---:|---:|---:|---:|
| 两场景平均 F-score@5cm ↑ | 28.7561% | 32.5102% | 32.3522% | **33.5447%** |
| 两场景平均 Chamfer-L1 ↓ | 15.3266 cm | 13.1772 cm | 13.2347 cm | **12.1151 cm** |
| 三场景合并平移 ATE ↓ | 0.8268 m | 0.7684 m | 0.7690 m | **0.7019 m** |
| 三场景合并旋转 RMSE ↓ | 22.8674° | 21.3330° | 21.3075° | **19.3241°** |

最后两列的实际边端点、变换矩阵、information 和权重均相同，唯一因素是缩放策略。这个直接对照的 F-score 为 +1.1925 pp、Chamfer 降低 8.4594%、平移/旋转误差降低 8.7206% / 9.3082%。相对 DPV，本轮候选平均 F-score 为 +4.7886 pp、Chamfer 降低 20.9540%。

几何聚合只包含两个 `0046` 场景；轨迹聚合包含 `0000_00` 和两个 `0046` 场景，按有效 GT 位姿数合并平方误差后开方，共 10,149 位姿。历史回环实验单独使用四场景 11,879 个有效 GT 位姿和三个 GT mesh，其均值不能与本表混用。完整推理帧没有因此被删减，差别来自评价所需的有限 GT 和候选可用性。

表中主几何指标使用共同 admitted frames 的 GT 深度支持所确定的参考表面：像素步长 8，支持距离 8 cm，评价下采样 3 cm。跨组公共参考 mask 和 surface SHA 已核对一致。完整 GT mesh 的次要评价及逐帧相对误差保留在外部原始 JSON，未用更有利的口径替换主结果。

## 几何门为什么仍拒绝，以及它自身的局限

| 场景 | 占用体素比 | 最小 extent 比 | 旧厚度比 | 旧层冲突比 | 旧安全 / 双改善门 |
|---|---:|---:|---:|---:|---|
| scene0000_00 | 0.9903 | 0.9760 | 1.0238 | 1.0040 | PASS / FAIL |
| scene0046_00 | 0.9588 | 1.0000 | 0.8436 | 0.9245 | PASS / FAIL |
| scene0046_01 | 0.9570 | 0.9452 | 1.0026 | 0.9102 | PASS / FAIL |

沿用规则要求厚度与层冲突都改善至少 10%，上述三个候选均未满足。不过本轮独立审计发现，旧指标并非已经验证的物理误差真值：

1. 旧层冲突按世界 x-z 格子和 y 高度分箱，完美单个竖直墙也可得到 100%“冲突”；整体平移 1 cm 可以使合成双平面的分数从 0 变 1。六个真实 PLY 在几何完全不变的整体平移下，旧分数也波动 2.50%–3.97%。
2. 旧平面匹配不要求投影支持重叠；厚度又只统计 15 mm RANSAC 内点，可能排除较厚部分。不同策略还会选中不同 baseline 平面，不能把跨策略厚度比直接当成同一物理平面的改善。

所以，**“双改善失败”不能单独证明真实重建没有改善**；同样，换指标后得到更好数字也不能证明精度提高。本轮保留旧门用于结果可比性，并新增 `geometry_metrics_v3.py` 独立诊断：使用法线与欧氏距离识别候选近邻层，增加平面支持重叠检查，并统计共同投影支持带的法线分布宽度。新判据通过刚体变换不变性测试；真实家具薄结构、点密度和 RGB-D 共同可见性仍需辨别，故固定 `usable_for_promotion=false`，不接入自动提交门。

下一步需要用预先保留的深度帧验证重投影残差、遮挡与自由空间违反，同时在新场景校准配准和几何指标；随后再研究直接优化局部表面对齐的约束。这些属于尚未执行的后续实验，本轮没有通过降低阈值宣称完成优化。

## 如何使用 develop 中的实验选项

原 `configs/pose/unified_backend.yaml` 保持不变。新增三个显式 profile：

| 配置 | 配准恢复 | 缩放 |
|---|---|---|
| `configs/pose/unified_backend_refined.yaml` | 开启 | global |
| `configs/pose/unified_backend_smooth_local.yaml` | 关闭 | smooth_local |
| `configs/pose/unified_backend_refined_local.yaml` | 开启 | smooth_local |

在具备原融合入口依赖和 CLIP 缓存的环境内，从仓库根目录运行，替换下面输入路径；输出目录必须不存在：

```bash
PYTHONPATH=src python -m pose_pipeline run-unified \
  --manifest /path/to/frontend/tracked_manifest.json \
  --trajectory /path/to/dpv/trajectory.json \
  --config configs/pose/unified_backend_refined_local.yaml \
  --clip-download-root /path/to/clip-cache \
  --output /path/to/NEW-refined-local-run
```

上面的组合 profile 已验证加载，两个新增机制由集成测试覆盖。本轮真实全入口使用 `unified_backend_refined.yaml`；最终局部对照使用其已封存的实际 PGO 边回放，没有重新运行一次 CLIP / PnP 来混入召回随机性。不能将此记录写成已单独重跑组合 profile 的四场景完整入口。

受控回放入口：

```bash
PYTHONPATH=src python scripts/replay_verified_unified_backend.py infer \
  --run-root /path/to/sealed/fresh_scene0046_00 \
  --inputs /path/to/frozen-inputs.json \
  --scene scene0046_00 \
  --config /path/to/verified-local-settings.json \
  --output /path/to/NEW-verified-local-run
```

settings 的 `bounded.correction_scaling_policy` 为 `smooth_local`，其他设置必须与源运行一致；本次还传入 `source_sha256`，绑定源结果、输入、回环证据、图结果和后端报告。脚本验证 RGB-D 像素哈希、输入帧、DPV、源配准证据、实际 PGO 边及有效权重，不重新生成回环候选、不读取 GT。无边时返回显式 DPV no-op。封存后使用既有 `scripts/validate_unified_pose_replay.py evaluate` 进行独立评价。

全入口下游仍应读取 `unified_result.json` 中的 committed trajectory / final cloud。本轮回放目录额外保留 `candidate_trajectory.json` 和 `candidate_refusion/refused.ply` 供研究查看；它们是被拒绝的候选，不应被误用作自动提交结果。

## 验证与证据

17 个核心 pose pipeline 测试文件共 **126 passed、119 subtests passed、0 skipped、0 failed**，包括真实 Open3D 配准恢复、独立 PnP family、累计修正限制、局部缩放连续性/求解失败回退、同源边篡改拒绝及几何反例。没有声称全仓历史 CUDA/模型实验测试通过。

运行主机为 RTX 4060 Laptop 8 GiB。完整新推理使用既有 `candidate-v2-runtime-20260904`，回放和 GT 使用 `sgaligner` Python；没有更换硬件比较速度。四个组的源码快照合计 1,248 项文件哈希核验通过；最终运行源码快照 317 文件，包 SHA-256 为 `99ac3b51656fd0771c2749ce71c7b48f91738b72972dcfbf1a3977f0c2512960`。测试日志 SHA-256 为 `1f22794da0cd0d0e5aa3114481e195eae5e1d0df70f871c63cf74331c75b4c0e`。

完整证据不进入 Git，保存在：

- 本地 `/Users/wu/Desktop/wu/SGF-SGA/comparisons/unified_pose_followup_20260907/`。
- 远端 `/home/aidenwu/Documents/unified-pose-followup-20260907/`。
- `factorial_analysis.json`、`smooth_local_analysis.json`、`fresh_refined_analysis.json`、`verified_local_analysis.json` 与 CSV：逐场景指标和可比性核验。
- `factorial/`、`smooth_local/`、`fresh_refined/`、`verified_local/`：完整轨迹、融合 PLY、输入绑定和决定。
- `geometry_metric_audit/`、`registration_audit/`：旧几何指标反例、独立诊断和配准失败定位。
- `followup_source_and_artifact_integrity.json`、`full_artifact_integrity.json`、`core_test_execution.json`：源码、推理封存、下载与测试证据。

这轮支持继续研究局部修正，并提供了可复现的平均候选收益；要把它作为更高精度的默认结果，还需要解决场景间退化和可靠的几何验收。
