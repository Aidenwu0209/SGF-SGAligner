# developnew 语义实例建图：首轮尝试

**结论：SGF、SGA、实例关联及逐点标签导出已接通；标签质量未通过，保留实验分支。**

本轮从 develop@1cf90f7 创建独立分支，固定既有完整轨迹和 TSDF 几何，再回放原始 RGB-D。没有重新训练，也没有改变相机位姿、原点云坐标、法线或 RGB。每种数据集验证 1 个完整样例，共 5363 个唯一原始帧；这不是全部场景验证。

| 数据集 / 场景 | 完整帧数 | 点数 | 最终标签覆盖 | 预测实例数 | SGA 接受关联 | 几何保留 |
|---|---:|---:|---:|---:|---:|---|
| scannet / scene0050_00 | 4652 | 348841 | 18.64% | 1165 | 216 | 逐字段一致 |
| orbbec / scene_001 | 524 | 47585 | 34.67% | 282 | 97 | 逐字段一致 |
| 3rscan / 09582214-e2c2-2de1-956a-64d8da4ba7cc | 187 | 56043 | 35.41% | 34 | 3 | 逐字段一致 |

标签覆盖指同时具有可靠类别与实例 ID 的点占原地图点数的比例，剩余点保留为未知；它不是准确率。

## 地图入口

每个目录都有 map_labeled.ply（原 RGB + 整数 semantic_id / instance_id / 置信度）、map_semantic.ply、map_instance.ply、objects.json、classes.json、scene_graph.json、result.json、verification.json。灰色为未知或冲突区域。实例 ID 的作用域是单场景。

- **scannet / scene0050_00**：[标签地图](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/scannet/scene0050_00/map_labeled.ply) · [语义着色](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/scannet/scene0050_00/map_semantic.ply) · [实例着色](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/scannet/scene0050_00/map_instance.ply) · [预览](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/scannet/scene0050_00/preview.png) · [读回验证](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/scannet/scene0050_00/verification.json)
- **orbbec / scene_001**：[标签地图](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/orbbec/scene_001/map_labeled.ply) · [语义着色](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/orbbec/scene_001/map_semantic.ply) · [实例着色](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/orbbec/scene_001/map_instance.ply) · [预览](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/orbbec/scene_001/preview.png) · [读回验证](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/orbbec/scene_001/verification.json)
- **3rscan / 09582214-e2c2-2de1-956a-64d8da4ba7cc**：[标签地图](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/3rscan/09582214-e2c2-2de1-956a-64d8da4ba7cc/map_labeled.ply) · [语义着色](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/3rscan/09582214-e2c2-2de1-956a-64d8da4ba7cc/map_semantic.ply) · [实例着色](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/3rscan/09582214-e2c2-2de1-956a-64d8da4ba7cc/map_instance.ply) · [预览](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/3rscan/09582214-e2c2-2de1-956a-64d8da4ba7cc/preview.png) · [读回验证](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/final/3rscan/09582214-e2c2-2de1-956a-64d8da4ba7cc/verification.json)

## 质量检查

GT 只在预测结束后的独立评价进程中读取。采用首个有效 GT 位姿进行坐标系对齐，不做 ICP 或尺度拟合。只评价距参考表面 5 cm 内、且 GT 类别名称与模型词表精确对应的点；以下不是官方 mIoU 或实例 AP。

| 场景 | 有参考且类别可比较的点 | 其中已有标签的点 | 已标注点语义正确率 | 未知也计错时正确率 | 条件实例纯度 |
|---|---:|---:|---:|---:|---:|
| scannet / scene0050_00 | 107827 | 20936 | 9.52% | 1.85% | 93.40% |
| 3rscan / 09582214-e2c2-2de1-956a-64d8da4ba7cc | 20618 | 7770 | 24.12% | 9.09% | 90.70% |

Orbbec 没有已验证的语义 / 实例 GT，本轮只做完整性、几何保留及可视化检查。高实例纯度不能掩盖碎裂：ScanNet 一个 GT 物体仍可能对应大量预测片段。因此本轮不认定标签质量通过，不替换 develop。

## 本轮改动和对照

- 使用 SGF 外部位姿接口：米制 T_world_camera 求逆后将平移转成毫米。相机标定和图像旋转与原融合共用读取逻辑。
- 全部样例使用 120 帧子图、30 帧重叠及相同参数，调用真实 SGF GraphPredictor 和官方 SGA pct/gat/rel checkpoint。SGA 与 SGF 使用各自现有 Python 环境，通过独立进程交接。
- SGA 对应需通过类别、双向表面重叠及配准残差检查；测得的位姿只作证据，不反馈到轨迹。
- 补上 SGF native instance / same part 合并，要求高置信度、类别一致及表面接触。保留每次合并记录。

  - scannet：实例数 1746 → 1165；标签覆盖 16.67% → 18.64%。实例数减少不直接等于实例准确率提高。
  - orbbec：实例数 733 → 282；标签覆盖 27.99% → 34.67%。实例数减少不直接等于实例准确率提高。
  - 3rscan：实例数 36 → 34；标签覆盖 35.31% → 35.41%。实例数减少不直接等于实例准确率提高。

- 相机上方向对照在 3RScan 上使条件语义正确率从约 24.29% 降到 4.68%，已经否决，未保留在分支入口。不能据此宣称坐标问题已解决，也没有按 GT 搜索最佳旋转。
- 数组化标签查找在完整 3RScan / Orbbec 点集上与旧导出器的标签、实例、置信度完全一致。

## 验证范围与复现

- 18 项代码测试通过，包含米 / 毫米和变换方向、冲突留未知、实例合并限制、SGF 静默关闭预测的失败处理及原 RGB-D 流程契约。
- 三个样例均读回验证了全部原始帧覆盖、原 XYZ / 法线 / RGB 逐字段一致、整数标签字段、物体表和场景图引用。原模型权重未修改。
- 3RScan 通过最终完整入口重跑；ScanNet、Orbbec 先完整回放 SGF / SGA，再复用冻结预测验证实例分组与导出。该分阶段验证没有重跑几何前端，记录绑定源文件哈希。
- 尚未验证跨扫描联合地图、非相邻回访身份一致性和全数据集泛化。语义错标、实例碎裂和未知区域仍是实际问题，不能把本轮称为完整质量验收。

运行说明：[SEMANTIC_MAPPING.md](/Users/wu/Desktop/wu/SGF-SGA/worktrees/developnew/docs/SEMANTIC_MAPPING.md)。完整源子图与日志保存在远端 `/home/aidenwu/Documents/sgf-sga-developnew-20260909`；本地最终地图统一位于本报告同目录的 `final/`。

测试：[18 项测试日志](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/evidence/tests_final_vectorized.log)；[导出器逐点一致性](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/evidence/exporter_parity.json)；[结构化汇总](/Users/wu/Desktop/wu/SGF-SGA/comparisons/developnew_semantic_20260909_v1/SUMMARY.json)。
