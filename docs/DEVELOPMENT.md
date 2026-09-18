# 源码维护

Git 仓库只需要保存本项目脚本、文档、许可和补丁；`src/`、`build/`、`.venv/`、SDK、模型、日志及测试输出均可在本地生成。源码准备和编译步骤见 [BUILD.md](BUILD.md)。

上游固定为 `ggml-org/llama.cpp` 提交 `bdcbaaf6e7520b68c8c60ff724c67409970d70e1`。项目补丁包含模型图、CUDA 融合核、批量采样前缀、原生批处理程序及回归测试。这是独立维护的定制分支；尚未向上游提交。若将来准备上游贡献，应重新阅读该版本的 `AGENTS.md`/`CONTRIBUTING.md` 并遵守其贡献规则。

## 修改与导出补丁

先按 `prepare_source.py` 获得源码，直接在 `src/llama.cpp` 修改。导出工具比较 HEAD，包含 staged、unstaged 和显式白名单中的新文件，不自动提交或改动开发 checkout 的 index。新增目录不在白名单时，需同步更新 `scripts/export_patch.py` 的 `EXTRA_DIRECTORIES` / `EXTRA_FILES`，避免漏掉实现。

建立另一个干净基线，仅用于 `git apply --check`：

```powershell
git clone --no-checkout --no-hardlinks src/llama.cpp src/llama-baseline
git -C src/llama-baseline -c core.autocrlf=false checkout --detach bdcbaaf6e7520b68c8c60ff724c67409970d70e1
.\.venv\Scripts\python.exe scripts/export_patch.py --dry-run
.\.venv\Scripts\python.exe scripts/export_patch.py
```

基线必须是独立、没有修改和未跟踪文件的 checkout。导出工具使用 PATH 中的 Git，也可用 `--git` 指定；不依赖 `runtime/git`。仅在 PATH 无 Git 且存在旧本地副本时，才采用该兼容回退。

主补丁为 `patches/hy-mt2.patch`；配套 `hy-mt2.manifest.json` 记录固定提交、补丁 SHA256、每个修改文件的原始/规范 LF 哈希及只读 apply 检查结果。路径相对本仓库记录。只分发主补丁，不需要早期 `phase1-*` 快照。

导出后，用新的目标目录验证整个准备流程：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_source.py --repository src/llama-baseline --destination results/source-check
```

此命令从本地 Git 对象获取固定提交，因此可以离线执行；并非复制已经修改的工作文件。它检查完整修改列表与 35 个当前修改文件的哈希，并在新 checkout 的 `.git/hy-mt2-prepared.json` 保存补丁和最终 Git tree ID。修改文件数量随未来补丁更新，不是硬编码要求。

本次已在独立 `results/source-repro-check-3` 验证主补丁：35/35 修改文件匹配；补丁 SHA256 为 `ad38fb9f932201fe88b2397395526eb6ca451bece1f0370df7c78ece8b3b93bb`，准备后 Git tree 为 `fad56bd8e27d21e686fc874e6742ed113b8e41f0`。这项检查验证源码复现，不涉及重新编译或 GPU 测试。

## 更改后的验证

按修改范围运行 [BUILD.md](BUILD.md) 中相应 CUDA 算子测试和 `test-hymt-prefix`，再比较真实 JSONL 翻译输出与吞吐。数值和质量结论的适用范围见 [优化实现](OPTIMIZATIONS.md) 与 [性能报告](PERFORMANCE.md)。修改编译架构后还需核对 `bin/build-info.json` 与实际 cubin，不能仅改元数据扩展支持名单。

`scripts/validate_portable.py` 会安全解压发布 ZIP 到新的中文和空格路径，检查 SHA256，隔离 PATH/Python 环境并执行 GPU 批量/API 验收；它拒绝覆盖旧目录，保留失败现场。只在实际需要测试最终包时运行，不能用静态脚本检查替代真实搬迁验收。

旧开发 SDK、编译中间文件、旧实验结果、早期补丁和基线 checkout 不属于运行必需文件。清理前保留主补丁、manifest、当前发布模型和已验证的二进制/运行 DLL；不得把清理整个目录的命令混入模型下载、构建或运行脚本。
