import type { Settings } from "./types";

export const DEFAULT_SETTINGS: Readonly<Settings> = Object.freeze({
  profile: "auto",
  gpu: "",
  context: 2048,
  parallel: 0,
  ubatch: 0,
  cache: "q8_0",
  memoryPercent: 30,
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
  for (const [key, label, min, max] of [
    ["context", "每请求上下文", 256, 32768],
    ["parallel", "请求并发", 0, 128],
    ["ubatch", "Micro batch", 0, 8192],
    ["port", "API 端口", 1024, 65535],
    ["memoryPercent", "显存预算比例", 10, 100],
  ] as const) {
    const value = settings[key];
    if (!Number.isInteger(value) || value < min || value > max) {
      throw new Error(`${label}须填写 ${min}–${max} 之间的整数。`);
    }
  }
}
