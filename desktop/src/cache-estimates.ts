import type { Plan } from "./types";

// Block scale metadata is included: F16=2, Q8_0=34/32, Q4_0=18/32 bytes.
export function cacheSaving(plan: Plan | null, cache: string) {
  if (!plan) return null;
  const budget = plan.budget;
  const baseline =
    (budget.kv_elements_per_token_per_cache *
      budget.kv_context_tokens_rounded *
      plan.parallel *
      4) /
    1024 ** 2;
  const ratio = cache === "q8_0" ? 34 / 64 : cache === "q4_0" ? 18 / 64 : 1;
  const saved = baseline * (1 - ratio);
  const fixed = budget.estimated_total_mib - budget.kv_total_mib;
  return {
    mib: saved,
    kvPercent: (1 - ratio) * 100,
    totalPercent: (saved / (fixed + baseline)) * 100,
  };
}

export const performanceHelp =
  "相对 F16 的历史原生批处理吞吐损失：Q8 约 13.6%（RTX 5090、当前 NVFP4、256 并发）；Q4 约 14.6%（旧实验 NVFP4、128 并发）。两组条件不同，不能据此比较 Q8 与 Q4 的速度。并非 HTTP API 延迟预测，其他显卡和 Q4_K_M 没有相应实测；实际速度、质量请按业务验证。显存节约按相同并发和上下文计算，包含量化元数据；总预算包含安全余量。";
