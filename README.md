# A-Share Market-Manipulation Detection — Mixture-of-Experts with Cross-Market Transfer

基于混合专家模型与跨市场辅助迁移的 A 股证券操纵行为智能识别。

论文正文见 [`thesis.md`](thesis.md)。方法是三阶段框架：用加密货币市场的量价异常样本对 MoE 时序编码器做自监督预训练，再用教师模型生成的思维链做轻量蒸馏，最后在 A 股确权案例上做元数据感知路由与分层消融。

## 目录

| 路径 | 内容 |
| --- | --- |
| `code/model/` | MoE 编码器、预训练、微调、基线与评估 |
| `code/data_pipeline/` | 证监会处罚文书、A 股行情与加密货币数据采集 |
| `code/` | 实验入口与远程训练辅助脚本 |
| `data/` | 处理后的案例与特征 |
| `figures/` | 实验图 |
| `thesis.md` | 论文全文 |
| `开题报告.md` 等 | 开题、可行性与阶段记录 |

## 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/model/requirements.txt
pip install -r code/data_pipeline/requirements.txt
cp code/data_pipeline/.env.example code/data_pipeline/.env
```

远程脚本和处罚书提取不再在代码里保存密钥，需要时从环境变量读取：

- `AUTODL_SSH_PASSWORD`：AutoDL 实例 SSH 密码
- `GOOGLE_API_KEY` 或 `GEMINI_API_KEY`：处罚书信息提取

## 训练入口

```bash
python code/model/scripts/run_pretrain.py
python code/model/scripts/run_finetune.py
python code/data_pipeline/run_pipeline.py full
```

## 未放入本仓库的内容

- 模型权重（`*.pt`，单文件约 100MB，且有多份重复备份）
- AutoDL 打包备份（单个压缩包超过 GitHub 100MB 文件上限）
- 第三方论文 PDF、学校表格模板，以及与本课题无关的个人文档
