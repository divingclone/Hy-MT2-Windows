import type { Settings } from "./types";

export const DEFAULT_SETTINGS: Readonly<Settings> = Object.freeze({
  profile: "auto",
  gpu: "",
  context: 2048,
  parallel: 0,
  ubatch: 0,
  cache: "int8_per_token_head",
  memoryPercent: 75,
  apiKeyEnabled: true,
  logMode: "memory",
  port: 18080,
});

export function resetInference(settings: Settings): Settings {
  return {
    ...DEFAULT_SETTINGS,
    port: settings.port,
    apiKeyEnabled: settings.apiKeyEnabled,
    logMode: settings.logMode,
  };
}

export function validateDraft(settings: Settings) {
  if (!["auto", "fast", "quality", "compat"].includes(settings.profile) || !["int8_per_token_head", "bfloat16", "fp8_per_token_head"].includes(settings.cache)) throw new Error("模型或 KV 配置无效");
  for (const [key, label, min, max] of [
    ["context", "每请求上下文", 256, 32768],
    ["parallel", "请求并发", 0, 256],
    ["ubatch", "调度 token 预算", 0, 8192],
    ["port", "API 端口", 1024, 65535],
    ["memoryPercent", "显存预算比例", 10, 100],
  ] as const) {
    const value = settings[key];
    if (!Number.isInteger(value) || value < min || value > max) {
      throw new Error(`${label}须填写 ${min}–${max} 之间的整数。`);
    }
  }
}

// Upgrade old llama.cpp preferences without losing port/auth/log choices.
export function migrateSettings(saved: Partial<Settings> = {}): Settings {
  const result = { ...DEFAULT_SETTINGS, ...saved };
  result.profile = result.profile === "official" ? "auto" : result.profile;
  result.cache = ({ q8_0: "int8_per_token_head", q4_0: "int8_per_token_head", f16: "bfloat16" } as Record<string,string>)[result.cache] ?? result.cache;
  return result;
}
