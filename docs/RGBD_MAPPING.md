# 从原始 RGB-D 完整建图

`run-rgbd` 从 manifest 中的全部 RGB-D 帧开始计算，不要求已有 DPV 轨迹。
流程为传感器深度约束 DROID-W、SIFT/PnP 后的 SVD 测量关键帧图、固定关键帧的视觉回填、全帧 TSDF 融合。三个数据集使用同一套参数，不根据场景名称或 GT 选择算法。

原来的 `run-unified` 是已有轨迹的后端优化入口；它不能补回前端未输出的原始帧。需要完整原始建图时使用下面的新入口。

```bash
PYTHONPATH=src "$CPU_PYTHON" -m pose_pipeline run-rgbd \
  --manifest /path/to/raw_manifest.json \
  --output /path/to/new_scene_output \
  --provider-root /path/to/DROID-W \
  --gpu-python "$GPU_PYTHON" \
  --cpu-python "$CPU_PYTHON"
```

GPU Python 需要现有 DROID-W 的 Torch、CUDA 扩展和 `lietorch`；CPU Python 需要 NumPy、SciPy、OpenCV 和 Open3D。不同阶段使用独立进程。外部 provider 必须已有 `configs/droid_w.yaml` 和 `pretrained/droid.pth`，程序不会安装依赖或下载模型。验证环境及 provider 来源以 `dense/result.json` 中的配置、代码和 checkpoint SHA 为准。当前验证使用 GPU Torch 2.3/CUDA 12.1 与 CPU Open3D 0.17 的既有环境；不能把另一组依赖版本视为已复现。

输出目录必须不存在。主要产物：

- `refill/trajectory.json`：每个原始 frame ID 和 timestamp 对应的最终位姿，单位米，`T_world_camera`。
- `fusion/refused.ply`：全部原始帧的 TSDF 点云，体素 2 cm、SDF 截断 8 cm、深度截断 4.5 m。
- `mapping_result.json`：轨迹、点云、输入和代码来源及完整帧数。仅在所有阶段成功且产物一致时生成。
- `run_status.json`、`logs/`：每阶段状态、实际命令、耗时和失败原因；失败现场保留。

`completed` 表示完整执行，几何质量由独立评价报告说明。缺乏有效深度或视觉约束、过短序列、关键帧缓存用尽等仍可能失败。不会用单位位姿补齐缺失帧。固定空深度关键帧仅在不作为任何实际回填来源时保留，真实回填权重仍须来自有效传感器深度。

## 批量运行

准备显式列表，数据集和场景名必须与 manifest 一致：

```json
[
  {"key": "scannet/scene0050_00", "manifest": "/data/scene0050_00/raw_manifest.json"},
  {"key": "orbbec/scene_001", "manifest": "/data/scene_001/raw_manifest.json"}
]
```

```bash
PYTHONPATH=src "$CPU_PYTHON" scripts/run_rgbd_matrix.py \
  --matrix /path/to/scenes.json --output /path/to/new_batch \
  --provider-root /path/to/DROID-W \
  --gpu-python "$GPU_PYTHON" --cpu-python "$CPU_PYTHON"
```

逐场使用同一流程；某场失败会记录后继续下一场，不覆盖旧结果，不自动换算法。默认单 GPU、每阶段最多 7200 秒，可用 `--stage-timeout` 设置。`batch_complete.json` 列出全部完成和失败，不能据此直接宣称所有地图质量通过。

## 当前方案选择依据

既有 16 ScanNet 和 11 Orbbec 实验中，轨迹覆盖从 92.3% 提升到 100%。与原 develop 使用相同共同帧时，ScanNet 的场景平均 ATE 从约 0.494 m 降至 0.133 m；15 个有几何参考的场景平均 F@5cm 从 33.15% 提升至 54.02%，其中 12 个提高、3 个降低。相邻旋转误差平均增加约 0.039°，局部跳变仍存在。Orbbec 没有绝对 GT，不能据传感器一致性宣布绝对精度提升。

这些是选取 SVD 流程的历史实验结果，正式入口的全量验证须另行记录。已有 3RScan 两个样例虽得到全帧轨迹，其中 443 帧样例仍严重失真；不能把覆盖改善等同于全部 3RScan 已修复。验收按整体实际收益与剩余问题严重程度权衡，允许披露后保留的小幅退化。
