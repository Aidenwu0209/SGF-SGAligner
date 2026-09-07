# 三方案融合：Hybrid36、RGB-D PnP 与完整几何 Guard

后续受控消融、局部缩放实现和旧几何指标的有效性问题见 [第二轮报告](unified_pose_followup_zh.md)。以下保留首轮运行记录；其中“层冲突”“厚度”和“同物理平面”是旧实现的字段名称，不应直接理解为已验证的物理误差真值。

本次从 `main@ef175964a3e20fba2c1a1795599325f2bff71122` 的代码开始，在 `develop` 集成三份交接材料中的可组合部分。它们不是三个可以叠加精度百分比的模型：Candidate 是框架，PnP 提供独立视觉证据，Guard 决定结果是否提交。本配置是开发实验，精度提升须看实际提交的轨迹与最终点云，不能用被回滚的候选指标代替。

## 1. 实际融合方式

```text
DPV 完整逐帧 T_world_camera（米）
  → Hybrid36 距离 + CLIP RGB 召回，预算 36
  → RGB-D 独立正反 SIFT/EPNP-RANSAC
  → PnP 作为一个独立 solver family，与三维 FPFH 假设共同做唯一簇共识
  → ICP、空间支持、重叠、正反配准一致性
  → PnP 与最终精配准变换再次核对
  → 去重 / 每锚点最多两条 loop / 长跨度 loop 限权
  → Huber PGO
  → 对所有 anchor 使用同一个缩放系数，传播到完整逐帧轨迹
  → 全轨迹绝对修正与相邻修正审计
  → 同一 admitted-frame manifest、同 TSDF 参数完整重融合
  → 安全 + 实际改善均通过：提交候选轨迹和对应点云
  → 任一失败：复制原 DPV 字节，交付对应完整重融合点云
```

PnP 必须在旧三维共识拒绝之前提供独立证据。首轮把 PnP 放在三维接受之后，`scene0030_02` 的 36 对候选全部提前退出，PnP 完全没有参与。最终实现允许真正的 `rgbd_pnp` family 加入原共识算法，仍要求至少两个独立 family，仍使用原 `2.5° / 0.05 m` 共识门，且必须唯一簇。两个方向的 PnP 都属于同一 family，不能算两票；PnP 单独通过不能建立图边。没有恢复已淘汰的 single-family fallback。

默认三维几何配置仍是 `baseline_mutual_fpfh` 与 decision v2。继承的 V3、GNC、SE(3) 传播代码仅保留接口/诊断兼容性，融合配置不启用这些实验算法。Huber 的 information/confidence 当前仍用于诊断，真正生效的限权作用在残差乘子 `edge.weight`，不能称为已实现信息矩阵加权 Huber。

普通 `run --arm candidate` 的调用和默认算法配置兼容，但新增的相邻 correction 审计会更保守地拒绝部分结果；不能把它描述为旧 main 所有接受结果逐位不变。

## 2. 新增与保留的实现

| 文件 | 职责 |
|---|---|
| `src/pose_pipeline/visual_verification.py` | 可复用双向 PnP 估计、独立 family 证据、与三维精配准结果核对 |
| `src/pose_pipeline/geometry_backend.py` | 在原三维共识中接受可选 PnP family，保留后续几何门 |
| `src/pose_pipeline/bounded_backend.py` | 限权、冻结 degree 选边、同一 Huber 的逐边删除审计、完整轨迹缩放 |
| `src/pose_pipeline/runner.py` | 联合证据、输入哈希、完整几何提交门与字节回滚 |
| `src/pose_pipeline/unified.py` | 消费实际 YAML 配置，交付最终 committed trajectory 与 full-refusion PLY |
| `configs/pose/unified_backend.yaml` | 三方案融合的可执行开发配置 |
| `scripts/validate_unified_pose_replay.py` | 冻结 PnP 回环的后端消融；推理、GT 评价为两个独立进程 |

PnP 现在读取每帧相机内参、manifest 深度尺度、图像旋转约定；反方向独立做 SIFT ratio matching，并检查正深度。旧脚本硬编码 `depth / 1000`、单一相机参数以及复用单向匹配，不能作为通用实现。RGB-D 仍需上游已配准：缩放 RGB 不能替代两台相机的外参标定。

## 3. 本轮固定的配置

- DPV、admitted frames、TSDF `0.02 / 0.08 / 4.5 m` 固定。
- 保留 Huber 与 legacy linear/Slerp 传播，避免一次替换多个优化器。
- 每个 anchor 的 loop degree 至多 2；一般 loop 残差权重至多 1.5，跨度达到 anchor 范围 75% 时至多 1.0。
- 统一校正缩放候选为 `[1, 0.5, 0.25, 0.125, 0.0625]`，选择第一个通过完整轨迹运动审计的系数；不对单帧分别截断。几何门失败后直接回滚，不搜索 GT 指标，也不调整门限。
- 全轨迹绝对修正上限 `0.25 m / 5°`，相邻 correction 变化上限 `0.05 m / 2°`。绝对量按 `C_i = T_candidate_i × inverse(T_DPV_i)` 定义，是 pilot 门限，尚未跨设备标定。
- 完整重融合必须保持体素数比例 ≥0.80、每轴 robust extent ≥0.85、匹配平面倾角退化 ≤2°、厚度和层冲突均不恶化超过 10%；厚度和层冲突还必须分别改善至少 10%。
- LOO 可显式开启，使用同一 Huber 从头重算，检查未缩放的完整轨迹影响；删除后重新验证剩余边，不补入先前未检查的边。其计算成本较高，融合配置默认关闭，单独作为消融臂验证。
- 所有边被删除属于 no-op，必须回滚原字节，不能记为 `backend_correction_applied=true`。

## 4. 运行

在项目根目录，使用具备 NumPy、SciPy、OpenCV、Open3D、plyfile、scikit-learn、PyYAML、PyTorch 与 OpenAI CLIP 的 Python 环境。CLIP 权重放入显式缓存目录；本次使用既有固定 CLIP 提交与模型缓存，没有改动用户的全局 Python 环境。

```bash
PYTHONPATH=src python -m pose_pipeline run-unified \
  --manifest /path/to/frontend/tracked_manifest.json \
  --trajectory /path/to/baseline/trajectory.json \
  --config configs/pose/unified_backend.yaml \
  --clip-download-root /path/to/clip-cache \
  --output /path/to/NEW-unified-run
```

必须传 DPV replay 产生的 `tracked_manifest.json`，其帧号和顺序与轨迹完全一致。输出目录必须不存在。主要产物：

- `input_binding.json`：输入、帧表、配置绑定。
- `loop_evidence.json`：每对候选的 PnP 估计、三维共识、最终复核与 submap 哈希。
- `bounded_backend.json`：实际限权、保留/删除边、可选 LOO、缩放与运动审计；没有通过验证的 loop 时不产生此文件。
- `trajectory.json`：唯一用于下游的已提交轨迹。
- `unified_result.json`：接受/回滚原因、轨迹 SHA、`final_cloud`、点云 SHA 与完整融合帧数。
- 候选通过运动门后，`precommit_refusion/` 保留 baseline/candidate 完整重融合及几何比较。早期失败则 `committed_refusion/` 重融合 DPV，仍交付完整点云。

可视化或场景图下游应读取 `unified_result.json` 的 `final_cloud` 和 `committed_trajectory`，不能读取被拒绝的 proposed/candidate 文件。

## 5. 验证与效果边界

本轮既验证真实入口，也验证冻结回环的后端效果。两者需要区分：冻结 replay 没有重新证明 Hybrid36 和新版 PnP 的注册覆盖率。具体数值和最终决定见同目录的 `unified_pose_validation_20260907.json`；外部完整点云、日志、哈希和旧失败尝试保留在 `comparisons/unified_pose_fusion_20260907/`。

四场景像素审计逐一核对 **11,882 帧 / 23,764 张 RGB-D 文件**，与原封存 SHA 全部一致。推理阶段不读取 GT；只有候选、提交结果、点云及决定封存后，独立评价进程才读取 ScanNet GT。

### 冻结回环后的限权/缩放实测

| 场景 | 统一缩放 | 占用体素比 | 最小 robust extent 比 | 厚度比 | 层冲突比 | 安全/双改善 |
|---|---:|---:|---:|---:|---:|---|
| scene0000_00 | 0.0625 | 0.9898 | 0.9884 | 1.0013 | 1.0055 | PASS / FAIL |
| scene0030_02 | 0.5 | 0.9429 | 0.9618 | 1.0036 | 0.9229 | PASS / FAIL |
| scene0046_00 | 0.25 | 0.9663 | 0.9667 | 1.0066 | 0.9316 | PASS / FAIL |
| scene0046_01 | 0.25 | 0.9714 | 0.9646 | 1.0149 | 0.9581 | PASS / FAIL |

四场景都完成全帧重融合和绝对修正审计，安全门从旧 sidecar 的 3/4 提高到本轮 4/4。`scene0000_00` 的占用比由旧 sidecar 的 0.7188 提高到 0.9898，说明限权和缩放缓解了过度收缩。由于同时改变两项，不能将收益单独归于其中一项。

三个有 GT mesh 场景的**诊断候选**平均 F-score@5cm 从 34.0696% 到 41.2429%（+7.1733 pp），平均 Chamfer-L1 从 0.136353 m 到 0.111514 m。该增益小于历史无遮限 sidecar，体现了抑制大修正与纠正漂移之间的取舍。**实际 committed trajectory 全部逐字节回到 DPV，提交结果的精度没有改变。**

`scene0030_02` 的独立 LOO 臂从 2 条回环中删除 1 条，保留 1 条；最终仍未达到双改善门，回滚。没有把“删完边”或“只改善某一个指标”记为成功优化。

### 最终融合入口的四场景实跑

该组重新运行 Hybrid36、独立双向 PnP、新跨 family 共识、PGO 和完整重融合，使用最终 v2 源码快照；它与上面的冻结旧 PnP 回环实验不是同一组数据结果。

| 场景 | 完整帧数 | PnP 质量通过 / 36 | 几何通过边 / degree 后边 | 缩放 | 最终决定 |
|---|---:|---:|---:|---:|---|
| scene0000_00 | 5429 | 15 | 9 / 7 | 0.0625 | 安全通过，双改善失败，回滚 |
| scene0030_02 | 1732 | 4 | 0 / 0 | — | 无独立几何共识，回滚 |
| scene0046_00 | 2383 | 9 | 3 / 3 | 0.25 | 安全通过，双改善失败，回滚 |
| scene0046_01 | 2338 | 6 | 4 / 4 | 0.25 | 安全通过，双改善失败，回滚 |

四场景都交付了完整 committed trajectory 与完整融合 PLY，11,882 帧无缺帧、无 identity fallback。三个产生候选的场景安全门通过；无 loop 的场景直接重融合 DPV，没有伪造一项候选几何比较。四个最终轨迹和点云的 SHA 都与各自 DPV 基线一致。

两个有新候选且具备 GT mesh 的场景：`scene0046_00` 的诊断 F-score@5cm 为 23.8427%→27.7089%，Chamfer 为 0.179667→0.151749 m；`scene0046_01` 分别为 33.6695%→37.3115%、0.126866→0.111796 m。它们仍被几何改善门拒绝，不能声称已提交更高精度的结果。

最终核心验证为 **88 passed、109 subtests passed、0 skipped、0 failed**，运行在既有 RTX 4060 主机的隔离 Python 环境。本机为 87 passed、1 skipped；完整真实 refusion 在远端执行。本轮不宣称全仓所有历史实验或 CUDA 扩展测试通过。

融合是否提高精度，以提交结果为准。若候选没有通过上述门限，最终结果等于 DPV；实现完成、单元测试通过、轨迹误差下降、点云有限以及某项 GT 指标变好，均不能替代这个结论。

## 6. 后续针对精度的工作

这三份材料主要处理全局漂移、错误 loop 和提交可靠性，无法保证改善局部深度噪声造成的平面厚度。后续应分别处理两个实测瓶颈：

1. **注册覆盖率**：在独立标定集评估视觉与三维假设的误差分布、失败方向和重复结构，增加确实独立的对象/局部几何证据。不要在当前四场景上为了获得接受边而放宽共识门。
2. **最终局部几何**：固定 RGB-D 与全帧输入，单独检验深度质量、局部 RGB-D 细化及融合权重对厚度的影响，再与本后端组合。全局 loop 修正和局部表面去噪需要分别做消融。

`main` 的默认配置继续保持原状；新增入口与实测结果保存在 `develop`，便于继续校准和对照。
