# developnew：固定几何上的 SGF / SGA 语义实例地图

本分支从 `develop@1cf90f7` 创建。`run-rgbd` 负责完整轨迹和 RGB 几何；新增
`run-semantic` 使用同一份 RGB-D manifest、最终轨迹和 TSDF 点云，输出真正的
逐点语义类别、实例 ID，以及物体表和关系图。当前是实验入口，不代表三种数据集
全部场景的语义与实例质量已经通过。

首轮三数据集完整样例及失败结论见
[验证报告](SEMANTIC_MAPPING_VALIDATION_20260909.md)：功能和几何保留检查通过，
语义质量未通过，保留为实验分支。

## 流程和边界

1. 校验原融合 receipt 与 manifest、trajectory、PLY 的 SHA 完全对应。
2. 将米制 `T_world_camera` 求逆，再把平移转成毫米，交给 SGF 外部位姿接口。
   RGB-D 的旋转、尺寸和相机内参与原融合共用同一个读取函数，深度截断同为 4.5 m。
3. SGF 在 120 帧、重叠 30 帧的子图上运行真实 InSeg 和 GraphPredictor，保留
   原始局部分割 ID、原生实例 ID、类别置信度和关系预测。子图最后一段不丢帧。
4. 在相邻子图调用现有官方 SGAligner checkpoint 的 `pct/gat/rel` 路径。
   不提供 GT 属性，不读取旧预测缓存，不训练权重。SGF 与 SGA 使用各自既有
   Python 环境，以子进程衔接。SGA top-3 候选再经过类别一致、双向表面覆盖、
   ICP 残差及接近单位变换检查；通过后关联场景范围内的实例 ID。
   再接入 SGF 原生实例分组：只有原生实例 ID 相同、预测类别一致且置信度至少
   0.5、`same part` 关系置信度至少 0.7，并有至少 3 个点在 5 cm 内接触的片段
   才合并。合并会沿已接受的 SGA 对应传播，并保留每次合并的证据。
5. 将子图标签投到原 TSDF 顶点，要求距离不超过 4 cm、法线夹角不超过 45°、
   类别置信度至少 0.5。距离相近（5 mm 内）的不同实例竞争时输出未知标签。
   `0` 表示未知，不以最近点强行覆盖所有区域。

跨子图候选采用保守的一对一策略，尚未解决任意 split/merge、非相邻回访及多扫描
全局身份一致性。局部未关联片段会保留为独立预测实例，不能据此宣称每个真实
物体只有一个 ID。墙/地等类别仍按预测区域保留 ID。本阶段测得的 ICP 变换仅作
关联证据，不应用到相机轨迹或最终几何；后续若启用位姿反馈，需另外验证。

## 调用

```bash
PYTHONPATH=src:/path/to/sgf/build/python:/path/to/sgf/python_bindings/python \
  /path/to/sgf-python -m pose_pipeline run-semantic \
  --manifest /data/raw_manifest.json \
  --trajectory /results/refill/trajectory.json \
  --baseline /results/fusion/refused.ply \
  --model /path/to/sgf/traced \
  --relation-vocab /path/to/sgaligner/checkpoints/release/relationships.txt \
  --sga-python /path/to/sgaligner-python \
  --device cuda --output /results/new_semantic_output
```

`--baseline` 同目录必须有 `refusion_result.json`。输出目录必须不存在。SGA 权重
沿用现有 `inference.sgf_official.inference.OFFICIAL_SNAPSHOT`；运行记录保存其
SHA。SGF 原生桥、模型各文件和本模块也记录 SHA。失败保留现场，不自动切换
模型或替换缺失预测。

## 输出

- `map_labeled.ply`：保留原 XYZ、法线、RGB，新增 `semantic_id`、`instance_id`、
  `semantic_confidence`。保存后逐字段读回检查原几何与颜色完全一致。
- `map_semantic.ply` / `map_instance.ply`：类别着色 / 实例着色预览，灰色为未知。
- `classes.json` / `objects.json`：类别字典、实例点数、中心与包围盒。
- `scene_graph.json`：最终可见实例之间的关系，以及局部到全局 ID 映射。
- `submap_*/`：真实 SGF 点云、预测图、帧列表、ID 对应。
- `sga_*.json` / `.log`：实际模型匹配、每个关联的接受/拒绝原因和配准证据。
- `instance_grouping.json`：SGF 原生同实例片段合并证据；子图 `global_ids.json`
  记录 SGA 关联后的 ID，最终合并后的 ID 以 `scene_graph.json` 为准。
- `result.json` / `status.json`：完成帧数、标签覆盖率、关联数、输入与模型来源。

`completed` 只表示运行完成。标签覆盖率不是语义准确率或实例 AP。先在 ScanNet
0050、Orbbec、3RScan 完整样例验证接线与几何保留，再检查错误类别、碎片、误合并
和未知区域。没有 GT 的场景不能报告绝对语义精度。

可用 `scripts/render_semantic_preview.py RESULT_DIR` 生成相同采样点、相同视角的
RGB / 语义 / 实例预览。`scripts/evaluate_semantic_map.py --help` 提供独立的
预测后诊断：首个有效 GT 位姿对齐、不做 ICP 或尺度拟合，仅统计距参考表面
5 cm 内、且 GT 类别名称在模型词表中的点。它会报告未知区域、条件语义正确率、
实例纯度与碎片数，**不等同于官方 mIoU/AP**，也不能用高实例纯度掩盖碎裂。

已有本入口生成的完整 SGF / SGA 结果时，可运行
`python -m pose_pipeline.semantic_instances --source SOURCE --output NEW_OUTPUT`
单独验证实例分组和标签导出。该模式复用冻结的预测，**不是重新运行前端或 SGF**；
记录会绑定源子图、模型运行记录和分组代码哈希。
