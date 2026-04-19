# DICOM–STL 对齐原型

轻量 PoC：读 DICOM、读 STL、Open3D 可视化、键盘手动刚体初对齐、ICP、配准前后叠加对比。**非临床、非生产**，仅供算法与工程验证。

## 环境

- Python 3.10+（推荐 3.12）
- 依赖：`pip install -r requirements.txt`

## 快速运行

```bash
# 默认：单层 WG04 CT + Stanford Bunny（演示用，解剖不对应）
python main.py

# 同源配对：多层 GE CT + 由同一体生成的表面 STL（仓库已带 sample_data/paired_ge_ct）
python main.py --preset paired-ge

# 无界面自检
python main.py --preset paired-ge --skip-manual --skip-preview
```

常用参数：`--dicom`（文件或目录）、`--stl`、`--skip-manual`、`--skip-preview`、`--no-auto-scale`、`--mesh-points`、`--max-volume-points`。

## 数据脚本

| 脚本 | 作用 |
|------|------|
| `scripts/download_sample_data.py` | 拉取 WG04 单层 DICOM + Bunny OBJ→STL |
| `scripts/download_paired_ge_ct.py` | 从 GitHub 拉取 `dcm_qa_ct` GE 28 层 DICOM，并本地生成同源 `surface_from_volume.stl` |

配对说明见 `sample_data/paired_ge_ct/README.txt`。

## 测试

```bash
python -m unittest discover -s tests -v
```

## 许可与数据

- 代码按项目需要自行标注许可。
- `dcm_qa_ct` 数据见原仓库许可；WG04 测试体、Stanford Bunny 等见各自来源。
