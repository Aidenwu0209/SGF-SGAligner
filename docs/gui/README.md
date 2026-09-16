# developnew 扫描 GUI

GUI 调用当前源码的 `run-sam3`，不是旧 `develop@1cf90f7` 的独立程序。相机采集与实时几何预览来自旧 GUI；停止后依次完成新轨迹/几何、SAM3、融合、可选 VLM 命名，再可选执行最新未知点补全。语义不是实时显示，不自动包含冻结证据的 P2；SGA 网络没有在此路径执行。

## ssh33 启动

已部署源码目录 `/home/aidenwu/Documents/SGF-SGAligner-developnew-gui-20260916`，沿用已存在的 Python 环境与权重。旧 GUI 目录不修改。

在 ssh33 桌面终端：

```bash
cd /home/aidenwu/Documents/SGF-SGAligner-developnew-gui-20260916
bash scripts/run_developnew_gui.sh
```

浏览器打开 `http://127.0.0.1:8765`。相机连接 ssh33。终端 Ctrl+C 关闭服务。

从 Mac 使用：

```bash
ssh -L 8765:127.0.0.1:8765 aidenwu@100.72.138.33 \
  'cd /home/aidenwu/Documents/SGF-SGAligner-developnew-gui-20260916 && bash scripts/run_developnew_gui.sh --no-browser'
```

Mac 浏览器打开同一地址；终端保持运行。服务只监听回环地址。

## 操作

1. 选择串行/阶段并行和已配置模型。默认串行 + Qwen NF4 为显存较保守设置；不宣称最优精度。
2. 默认开启多视角未知点补全，此附加阶段固定使用 Qwen NF4 + SAM3；无论上面的命名模型是什么，都会单独运行。要完全禁用 VLM，选择 `none` 并取消补全。
3. 开始扫描，缓慢移动；停止并生成最终地图。采集与预览可取消，保留原始数据。
4. 完成后切换语义、实例、原色；保存 PLY（包含 semantic_id / instance_id）和轨迹。

输出在 `/home/aidenwu/Documents/SGF-developnew-GUI-scans/scan_*`，每次独立目录。`mapping.log` 和 `pipeline/*.log` 为诊断日志，`pipeline/GUI_RESULT.json` 指向最终结果。预览最多显示10万点，下载PLY保留完整点数。refinement只填有直接多帧支持的未知点，几何和已有实例不变。

`configs/runtime.ssh33.local.json` 是主机配置，不提交到Git；可通过 `SGF_SEMANTIC_RUNTIME` 指向其他配置。Python与模型目录不随源码分发。`SGF_SCAN_OUTPUT` 可更改输出位置。端口冲突时追加 `--port 8767`。

## 验证命令

仅检查部署入口，不连接相机、不运行GPU：

```bash
cd /home/aidenwu/Documents/SGF-SGAligner-developnew-gui-20260916
bash scripts/run_developnew_gui.sh --help
```

真实录制数据回放（启动后在网页点开始）：

```bash
bash scripts/run_developnew_gui.sh --no-browser --port 8766 \
  --replay /home/aidenwu/Documents/SGF-SGA-experiments/model_scale_deployment_20260915_v1/pipeline/inputs/orbbec.json \
  --fps 30 --max-frames 120
```

回放输入必须已做RGB-D注册；这里使用已有Orbbec注册输入。不能把原生未注册ScanNet RGB简单冒充注册数据。
