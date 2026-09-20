# 性能与质量

当前后端为 Windows 原生 vLLM，默认 NVFP4 W4A4 + INT8 KV。

README 主对照使用未修改的官方 llama.cpp、腾讯官方 Q4_K_M 与本项目 vLLM，双方采用 aiohttp 异步连接池，通过非流式翻译 API 测量完整响应；见 [API 优化与官方对照](API_OPTIMIZATION.md)。[早期同步客户端结果](OFFICIAL_API_BENCHMARK.md)、下列核心引擎及历史 SSE 指标保留原口径，不与新 API 数值混用。

- [构建与开发](BUILD.md)
- [KV 与核心吞吐对照](VLLM_KV.md)
- [未量化模型参照评测](TEACHER_FIDELITY.md)
- [历史 llama.cpp 文档](../archive/llama-cpp/docs/PERFORMANCE.md)
