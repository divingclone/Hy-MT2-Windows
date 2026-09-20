export interface Settings {
  profile: string;
  gpu: string;
  context: number;
  parallel: number;
  ubatch: number;
  cache: string;
  port: number;
  memoryPercent: number;
  apiKeyEnabled: boolean;
  logMode: "memory" | "file" | "off";
}
export interface Preferences {
  settings: Settings;
  closeToTray: boolean;
  minimizeToTray: boolean;
  silentAutostart: boolean;
  startService: boolean;
}
export interface Failure {
  message: string;
  hint?: string;
  code?: string;
}
export interface GPU {
  index: number;
  uuid: string;
  name: string;
  total_memory_mib: number;
  free_memory_mib: number;
  driver_version: string;
  compute_capability: string;
  reclaimable_memory_mib: number;
  reclaim_is_estimate: boolean;
  redeploy_available_mib: number;
}
export interface Model {
  profile: string;
  filename: string;
  size_bytes: number;
  quantization: string;
  installed: boolean;
  downloaded: number;
  downloadable?: boolean;
}
export interface Inventory {
  gpus: GPU[];
  models: Model[];
  data_dir: string;
  model_dir: string;
  model_warnings?: string[];
  runtime_dir: string;
  gpu_error?: Failure;
}
export interface Plan {
  settings: Settings;
  profile: string;
  gpu: GPU;
  parallel: number;
  context: number;
  ubatch: number;
  cache: string;
  budget: {
    estimated_total_mib: number;
    model_reserve_mib: number;
    kv_total_mib: number;
    kv_token_capacity: number;
    full_context_sequences: number;
    workspace_reserve_mib: number;
    safety_margin_mib: number;
    free_memory_mib: number;
    memory_percent: number;
    memory_limit_mib: number;
    usable_memory_mib: number;
    note: string;
    kv_context_tokens_rounded: number;
    kv_elements_per_token_per_cache: number;
  };
}
export interface Task {
  phase: string;
  port?: number;
  pid?: number;
  profile?: string;
  plan?: Plan;
  error?: Failure;
  message?: string;
}
export interface Status {
  service: Task;
  download: Task;
  progress: Record<string, number>;
  portable: boolean;
}
