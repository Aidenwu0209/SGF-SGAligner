# High-recall pose preset 全量验证报告

结论：**不通过，不应提升为默认或推荐 preset。** 代码继续保持 opt-in；`main` 未修改。

## 实验身份

- 仓库：`Aidenwu0209/SGF-SGAligner`
- 分支/提交：`develop@5cc5712ab8e9a7122e882f8e4dab208948c7885a`
- 参数：`maximum_loop_pairs=120`、`high_leverage_loop_min_span_fraction=1.0`、`high_leverage_loop_weight_cap=0.3`
- 权威主机：`100.72.138.33`
- 推理阶段不读取 GT；ScanNet/3RScan GT 只由独立评估过程读取。

## 可以跑多少帧

| 数据 | 序列/扫描 | 输入帧 | 有效 pose | 覆盖率 | 回环 | 主要结果 |
|---|---:|---:|---:|---:|---:|---|
| ScanNet | 16/16 | 37,296 | 35,047 | 93.970% | 38 | held-out ATE 平移/旋转改善 31.32%/33.66%；最终几何改善仅 1/16 |
| 3RScan | 179/179 | 41,487 | 139 | 0.335% | 0 | 93 个扫描零 pose；official pair recall 为 0；candidate 有 1 个灾难性接受 |
| Orbbec | 5/5 | 4,126 | 4,126 | 100% | 5 | 0/5 达到主要改善，3/5 通过安全门 |
| 合计 | 200 个序列/扫描 | 82,909 | 39,312 | 47.416% | 43 | 合计覆盖率被 3RScan 前端失败主导，不宜跨数据集直接比较 |

Orbbec 本轮复用了同一批完整 DPV 无 GT 轨迹，再跑 high-recall 后端和 refusion；因此 4,126/4,126 是同输入后端 A/B，而不是重新估计一次前端。

## ScanNet

- 16 个场景全部完成，12 个应用了 correction，4 个 fail-closed no-op。
- 实际 11 个 held-out 场景：
  - 平移 ATE 改善 31.32%，paired bootstrap 95% CI `[13.54%, 45.79%]`。
  - 旋转 ATE 改善 33.66%，paired bootstrap 95% CI `[13.81%, 48.43%]`。
- 0 个灾难性评估边，但有 1 条 accepted edge 因对应 GT pose 非有限/缺失而不可评估。
- 10/16 通过几何安全门，只有 `scene0046_00` 同时达到重影冲突改善至少 10%。
- 典型失败：
  - `scene0000_00`：pose 明显改善，但点数只保留 78.0%，几何安全失败。
  - `scene0030_02`：重影冲突改善 15.4%，但地面倾角增加 5.59°。
  - `scene0046_01`：地面倾角增加 13.61°。
  - `scene0046_02`：pose 平移/旋转改善 56.0%/57.4%，但重影冲突恶化 4.47%，平面厚度恶化 20.2%。

协议审计发现实际 split 是 5 个 development、11 个 held-out，而计划写的是 4/12。原因是 driver 已按 UUID hash 选择 4 个 development，本轮又显式加入 `scene0030_00`，形成第 5 个。结果没有被事后重分类；主结论按实际 11 个 held-out 报告。

## 3RScan

- 选择清单列出 180 个扫描，实际存在 179 个：156 validation、22 development、1 failure sentinel，共 41,487 帧。
- 86 个扫描产出至少一个 pose，93 个扫描为零 pose；累计仅 139 个 pose，单扫描最多 18 个，0 个扫描达到 80% 覆盖率。
- 主要前端失败是 metric scale 无法稳定提交：RGB-D-backed keyframe 数不足，或尺度从约 `5.54` 跳到 `0.37`，随后出现约 `6 m / 140°` 的不连续运动并被门控拒绝。
- 287 个组内 pair 中 90 个有文件进入推理，197 个因缺失/过小 refusion cloud 失败。
- 109 个 official reference pairs 中只有 28 个有完整推理输入；baseline/candidate 的 `5°/0.2 m` recall 都为 0。
- candidate 唯一接受的 official pair 为 `d7d40d50... -> d7d40d4e...`，评估误差 `8.16° / 2.25 m`，属于灾难性错误，3RScan gate 明确失败。

## Orbbec

- 五个完整序列均为 100% pose 覆盖。
- `fast_turn` 没有可验证回环，结果是严格 no-op。
- `leave_and_return` 的 layer-conflict 改善 24.39%，但点数只保留 71.32%，安全失败。
- `slow_table_loop` 的 layer-conflict 恶化 28.64%。
- `sgf_parameter_control` 因候选预算增加而新接受 1 条回环，但 layer-conflict 恶化 2.72%。
- 总体为 0/5 主要改善、3/5 安全，远低于要求的至少 4/5 改善且全部不显著恶化。

## 参数为什么没有达到预期

1. 候选预算从 36 增到 120，确实让 ScanNet 更多场景找到回环并改善 ATE。
2. `span_fraction=1.0` 只会压低精确首尾跨度的边。本轮 ScanNet 接受的 38 条回环中，实际被 `0.3` cap 影响的是 **0 条**；因此保护参数没有发挥作用。
3. ScanNet 的主要问题变成 pose graph correction 传播过强：轨迹误差下降，但姿态倾斜、点数损失或表面厚度恶化。
4. 3RScan 的瓶颈发生在 DPV metric-scale/数据适配阶段，远早于 PAGOR/G3Reg/TEASER++ 后端，调 loop 参数无法解决。
5. Orbbec 没有显示出跨序列泛化：增加候选只对 1 个序列新增回环，而且该序列反而略差。

另外，同配置的 `scene0030_00` 本轮 DPV 输出 2,271 个 pose，先前串行试验为 2,277 个，轨迹 SHA 也不同。固定 seed 仍不能保证并发 CUDA 前端逐位复现；本轮 A/B 内部输入相同，但运行时间和跨次 SHA 不应被当作确定性基准。

## 兼容性与测试

- x86 权威环境 focused pose suite：27 个 unittest + 2 个 public-driver 函数测试，共 29 个通过。
- 全仓库 legacy `unittest discover`：运行 631 个测试，34 个 collection/environment error、2 个 skip，因此不能报告为全绿。错误主要来自该浅克隆未安装 `pytest`、缺少冻结 outputs/checkpoint，以及旧 pilot 模块/API 无法导入；本次 focused pose suite 没有失败。
- ARM64 (`100.105.135.18`)：16 个纯 SE(3)/兼容图/门控/合同测试通过；41/41 帧 Orbbec journal 成功生成 `rgbd_sequence_manifest.v1`，`gt_at_inference=false`。
- ARM 节点没有安装 Open3D、TEASER++、pyGCRANSAC，因此没有运行重依赖矩阵，剩余磁盘保持约 5 GB。
- 未发现 OOM、非有限输出矩阵、identity fallback 或 refusion 融合帧漏覆盖。

## 时间（资源竞争下，仅供容量估计）

- Orbbec cached-backend + refusion：约 9.6 分钟。
- ScanNet 全矩阵墙钟约 90 分钟；16 个前端串行时间合计约 94.3 分钟，平均约 151.7 ms/输入帧。多个 CUDA worker 并发，因此不是性能基准。
- 3RScan 序列矩阵约 87.8 分钟；pair inference/evaluation 约 2 分钟。

## 决策

- 保留 high-recall preset 为明确的 opt-in 实验；安全默认值不变。
- 不推到 `main`，不推荐用于当前 demo/Orbbec 生产路径。
- 下一轮优先级应是：先修复 3RScan 前端尺度适配；再把 high-leverage span 阈值降到能实际触发的区间，并加入最大 correction/倾角代理门；最后重新做同输入全链路 refusion 验证，而不是继续单纯增加候选数。
