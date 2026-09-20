import type { Plan } from "./types";

// Same cached token capacity, including FP32 per-token/head scales.
export function cacheSaving(plan: Plan | null, cache: string) {
  if (!plan) return null;
  const baseline = plan.budget.kv_token_capacity * 65536 / 1024 ** 2;
  const ratio = cache === "bfloat16" ? 1 : 132 / 256;
  const saved = baseline * (1 - ratio);
  const fixed = plan.budget.estimated_total_mib - plan.budget.kv_total_mib;
  return { mib: saved, kvPercent: (1 - ratio) * 100, totalPercent: saved / (fixed + baseline) * 100 };
}
export const performanceHelp = "默认动态 INT8 KV；相同 token 容量下缓存减少 48.44%。RTX 5090 核心测试中，32/256 并发吞吐相对 BF16 配置分别变化 −6.0%/+5.9%，整任务显存约减少 26%/32%。这是特定负载实测，不是 API 延迟保证。KV 是共享 token 池，不为每个请求预留完整上下文。";
