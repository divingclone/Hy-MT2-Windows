<script lang="ts">
  import { validateDraft, DEFAULT_SETTINGS, resetInference } from "./config";
  import Help from "./Help.svelte";
  import { cacheSaving, performanceHelp } from "./cache-estimates";
  import { onMount } from "svelte";
  import { invoke, isTauri } from "@tauri-apps/api/core";
  import { getVersion } from "@tauri-apps/api/app";
  import { listen } from "@tauri-apps/api/event";
  import { enable, disable, isEnabled } from "@tauri-apps/plugin-autostart";
  import { writeText } from "@tauri-apps/plugin-clipboard-manager";
  import { open, confirm } from "@tauri-apps/plugin-dialog";
  import type {
    Settings,
    Preferences,
    Inventory,
    Plan,
    Status,
    Failure,
    Model,
  } from "./types";

  const defaults = DEFAULT_SETTINGS;
  let settings = $state<Settings>({ ...defaults });
  let prefs = $state<Omit<Preferences, "settings">>({
    closeToTray: false,
    minimizeToTray: true,
    silentAutostart: false,
    startService: false,
  });
  let tab = $state("service");
  let inventory = $state<Inventory>({
    gpus: [],
    models: [],
    data_dir: "",
    model_dir: "",
    runtime_dir: "",
  });
  let runtimeState = $state<Status>({
    service: { phase: "stopped" },
    download: { phase: "idle" },
    progress: {},
    portable: false,
  });
  let plan = $state<Plan | null>(null);
  let planError = $state<Failure | null>(null);
  let error = $state<Failure | null>(null);
  let toast = $state("");
  let initialized = $state(false);
  let busy = $state(false);
  let auto = $state(false);
  let estimating = $state(false);
  let logs = $state("");
  let version = $state("0.1.3");
  const contextOptions = [512, 1024, 2048, 4096, 8192, 16384, 32768];
  let customContext = $state(false);
  const contextChoice = $derived(
    customContext || !contextOptions.includes(settings.context)
      ? "custom"
      : String(settings.context),
  );
  let update = $state<{ version: string; body?: string } | null>(null);
  let updating = $state(false);
  let updateBytes = $state(0);
  let updateTotal = $state(0);
  let estimateId = 0;
  let toastTimer: ReturnType<typeof setTimeout>;
  const phases: Record<string, string> = {
    stopped: "服务未启动",
    starting: "正在启动",
    ready: "服务运行中",
    error: "需要处理",
    downloading: "正在下载",
    verifying: "正在校验",
    complete: "已完成",
    idle: "等待操作",
  };
  const running = $derived(runtimeState.service.phase === "ready");
  const active = $derived(
    ["starting", "ready"].includes(runtimeState.service.phase),
  );
  const downloading = $derived(
    ["starting", "downloading", "verifying"].includes(
      runtimeState.download.phase,
    ),
  );
  const baseUrl = $derived(
    `http://127.0.0.1:${runtimeState.service.port ?? settings.port}/v1`,
  );
  const displayedPlan = $derived(plan);
  const dirty = $derived(
    active &&
      JSON.stringify(settings) !==
        JSON.stringify({ ...defaults, ...runtimeState.service.plan?.settings }),
  );
  const currentAuth = $derived(
    runtimeState.service.plan?.settings.apiKeyEnabled ?? true,
  );
  const logMode = $derived(
    runtimeState.service.plan?.settings.logMode ?? settings.logMode,
  );
  function savingLabel(cache: string) {
    const saving = cacheSaving(plan, cache);
    return saving
      ? `省 ${gib(saving.mib)} GiB · 总预算 −${saving.totalPercent.toFixed(1)}%`
      : "显存估算待计算";
  }
  const currentGPU = $derived(
    displayedPlan?.gpu ??
      inventory.gpus.find((gpu) => !settings.gpu || gpu.uuid === settings.gpu),
  );
  const serviceGPU = $derived(
    active ? (runtimeState.service.plan?.gpu ?? currentGPU) : currentGPU,
  );
  const modelName = (profile?: string) =>
    profile === "fast"
      ? "Hy-MT2 · NVFP4"
      : profile === "official"
        ? "Hy-MT2 · Q4_K_M"
        : "自动选择";
  const gib = (mib: number) => (mib / 1024).toFixed(2);
  const fileSize = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  function failure(value: unknown): Failure {
    if (typeof value === "string") {
      try {
        const result = JSON.parse(value);
        if (result.message) return result;
      } catch {}
      return { message: value };
    }
    return { message: String(value) };
  }
  function notify(message: string) {
    toast = message;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (toast = ""), 3500);
  }
  async function perform(fn: () => Promise<unknown>) {
    busy = true;
    error = null;
    try {
      await fn();
    } catch (e) {
      error = failure(e);
    } finally {
      busy = false;
    }
  }
  const rpc = <T,>(action: string, args: unknown = {}) =>
    invoke<T>("rpc", { action, args });
  async function refresh() {
    inventory = await rpc<Inventory>("inventory");
  }
  async function poll() {
    const previous = runtimeState.download.phase;
    runtimeState = await invoke<Status>("status");
    if (
      runtimeState.download.phase !== previous &&
      ["complete", "error"].includes(runtimeState.download.phase)
    )
      await refresh();
  }
  async function save() {
    validateDraft(settings);
    await invoke("preferences", {
      value: { ...prefs, settings: { ...settings } },
    });
  }
  async function estimate(args: Settings) {
    const id = ++estimateId;
    estimating = true;
    try {
      validateDraft(args);
      const next = await rpc<Plan>("estimate", args);
      if (id === estimateId) {
        plan = next;
        planError = null;
      }
    } catch (e) {
      if (id === estimateId) {
        plan = null;
        planError = failure(e);
      }
    } finally {
      if (id === estimateId) estimating = false;
    }
  }
  $effect(() => {
    const snapshot = { ...settings };
    if (!initialized) return;
    ++estimateId;
    plan = null;
    planError = null;
    estimating = true;
    const timer = setTimeout(() => {
      void estimate(snapshot);
    }, 500);
    return () => clearTimeout(timer);
  });
  async function start() {
    await save();
    await invoke("task_start", { kind: "service", args: { ...settings } });
    await poll();
  }
  async function stop() {
    await invoke("task_stop", { kind: "service" });
    await poll();
    await refresh();
  }
  async function restart(rotateKey = false) {
    validateDraft(settings);
    await invoke("redeploy", { args: { ...settings, _rotate_key: rotateKey } });
    await save();
    await poll();
    notify(
      rotateKey ? "正在生成新 Key 并重新部署" : "检查通过，正在应用新配置",
    );
  }
  async function download(model: Model) {
    await invoke("task_start", {
      kind: "download",
      args: { profile: model.profile },
    });
    await poll();
  }
  async function importModel(model: Model) {
    const path = await open({
      title: `导入 ${modelName(model.profile)}`,
      filters: [{ name: "GGUF 模型", extensions: ["gguf"] }],
      multiple: false,
      directory: false,
    });
    if (typeof path === "string") {
      await invoke("task_start", {
        kind: "import_model",
        args: { profile: model.profile, path },
      });
      await poll();
    }
  }
  async function remove(model: Model) {
    if (
      await confirm(
        `删除 ${modelName(model.profile)} 及未完成下载？${runtimeState.portable ? "只删除本免安装版目录中的模型，其他目录原文件不会删除。" : "其他安装版也将无法使用这份共享模型，旧目录原文件不会删除。"}之后可重新下载。`,
        { title: "删除模型", kind: "warning" },
      )
    ) {
      await rpc("remove_model", { profile: model.profile });
      await refresh();
    }
  }
  async function copy(text: string) {
    await writeText(text);
    notify("已复制");
  }
  function restoreDefaults() {
    settings = resetInference(settings);
    customContext = false;
    notify(
      active
        ? "已恢复默认推理配置，重新部署后生效"
        : "已恢复默认推理配置，保存或启动服务后生效",
    );
  }
  async function toggleAuto() {
    if (auto) await disable();
    else await enable();
    auto = await isEnabled();
  }
  async function checkUpdate() {
    update = await invoke("check_update");
    if (!update) notify("已经是最新版本");
  }
  async function installUpdate() {
    if (
      !(await confirm("更新会停止推理服务和模型下载，然后重启应用。继续？", {
        title: "安装更新",
        kind: "warning",
      }))
    )
      return;
    updating = true;
    updateBytes = 0;
    try {
      await invoke("install_update");
    } finally {
      updating = false;
    }
  }
  onMount(() => {
    if (!isTauri()) {
      error = {
        message: "请通过 HyMT 桌面程序使用此界面。",
        hint: "开发运行：在 desktop 目录执行 npm run tauri dev。",
      };
      return;
    }
    let disposed = false;
    let polling = false;
    let unlisten: (() => void) | undefined;
    let unlistenPrefs: (() => void) | undefined;
    let lastEstimate = 0;
    const timer = setInterval(async () => {
      if (!initialized || disposed || polling) return;
      polling = true;
      try {
        const oldPhase = runtimeState.service.phase;
        await poll();
        if (
          !busy &&
          !estimating &&
          (Date.now() - lastEstimate > 10000 ||
            oldPhase !== runtimeState.service.phase)
        ) {
          lastEstimate = Date.now();
          await refresh();
          await estimate({ ...settings });
        }
      } catch (e) {
        if (!disposed) error = failure(e);
      } finally {
        polling = false;
      }
    }, 1000);
    void perform(async () => {
      const saved = await invoke<Partial<Preferences>>("preferences");
      settings = { ...defaults, ...saved.settings };
      prefs = {
        ...prefs,
        ...Object.fromEntries(
          Object.entries(saved).filter(([key]) => key !== "settings"),
        ),
      };
      auto = await isEnabled();
      version = await getVersion();
      await refresh();
      await poll();
      unlisten = await listen<{ chunk: number; total?: number }>(
        "update-progress",
        ({ payload }) => {
          updateBytes += payload.chunk;
          updateTotal = payload.total ?? 0;
        },
      );
      unlistenPrefs = await listen<Preferences>(
        "preferences-changed",
        ({ payload }) => {
          prefs.closeToTray = payload.closeToTray;
        },
      );
      if (disposed) {
        unlistenPrefs();
        unlisten();
        return;
      }
      initialized = true;
    });
    return () => {
      disposed = true;
      clearInterval(timer);
      clearTimeout(toastTimer);
      unlisten?.();
      unlistenPrefs?.();
    };
  });
</script>

<svelte:head><title>HyMT · 翻译后端</title></svelte:head>

<div class="app-shell">
  <aside>
    <div class="brand">
      <div class="brand-mark">H<span>↗</span></div>
      <div><strong>HyMT</strong><small>本地翻译后端</small></div>
    </div>
    <div class="nav-label">工作空间</div>
    <nav aria-label="主导航">
      {#each [{ id: "service", icon: "◉", label: "服务概览" }, { id: "models", icon: "▦", label: "模型管理" }, { id: "settings", icon: "☷", label: "应用设置" }, { id: "logs", icon: "≡", label: "运行诊断" }] as item}
        <button
          class:chosen={tab === item.id}
          onclick={() => {
            tab = item.id;
            if (item.id === "logs")
              void perform(async () => (logs = await rpc<string>("logs")));
          }}><span class="nav-icon">{item.icon}</span>{item.label}</button
        >
      {/each}
    </nav>
    <div class="sidebar-bottom">
      <div class="local-badge">
        <span class="dot" class:online={running}></span>
        {running ? "本地 API 在线" : "本地运行 · 数据留在本机"}
      </div>
      <small
        >HyMT {version} · {runtimeState.portable ? "免安装版" : "桌面版"}</small
      >
    </div>
  </aside>

  <main>
    <header>
      <div>
        <div class="eyebrow">HYMT / LOCAL INFERENCE</div>
        <h1>
          {tab === "service"
            ? "服务概览"
            : tab === "models"
              ? "模型管理"
              : tab === "settings"
                ? "按你的方式运行"
                : "运行诊断"}
        </h1>
        <p>
          {tab === "service"
            ? "管理推理服务，为其他应用提供兼容 OpenAI 的 API。"
            : tab === "models"
              ? "按显卡选择模型，下载后即可离线使用。"
              : tab === "settings"
                ? "启动、后台运行与更新，集中设置。"
                : "查看服务日志，快速定位运行问题。"}
        </p>
      </div>
      <span class="platform-tag">WINDOWS · NVIDIA</span>
    </header>
    {#if error}<div class="notice danger" role="alert">
        <div>
          <strong>{error.message}</strong>{#if error.hint}<p>
              {error.hint}
            </p>{/if}
        </div>
        <button
          class="icon-button"
          aria-label="关闭提示"
          onclick={() => (error = null)}>×</button
        >
      </div>{/if}
    {#if dirty}<div class="notice pending" role="status">
        <div>
          <strong>配置有待应用的更改</strong>
          <p>
            重新部署会先检查配置，再停止旧服务并应用更改；期间 API 会短暂中断。
          </p>
        </div>
        <button
          class="primary"
          disabled={busy ||
            updating ||
            downloading ||
            runtimeState.service.phase === "starting"}
          onclick={() => perform(() => restart())}
          >{busy ? "正在处理…" : "重新部署 · 应用更改"}</button
        >
      </div>{/if}
    {#if runtimeState.service.phase === "error" && runtimeState.service.error}<div
        class="notice danger"
        role="alert"
      >
        <div>
          <strong>推理服务未运行</strong>
          <p>{runtimeState.service.error.message}</p>
          <p>{runtimeState.service.error.hint}</p>
        </div>
        <button
          onclick={() => {
            tab = "logs";
            void perform(async () => (logs = await rpc<string>("logs")));
          }}>查看日志</button
        >
      </div>{/if}

    {#if tab === "service"}
      <section class="service-card">
        <div class="service-heading">
          <div class="status-symbol" class:online={running}>
            {running ? "↗" : "⏻"}
          </div>
          <div>
            <h2>
              {phases[runtimeState.service.phase] ?? runtimeState.service.phase}
            </h2>
            <p>
              {running
                ? `${modelName(runtimeState.service.plan?.profile)} · ${runtimeState.service.plan?.parallel} 并发`
                : runtimeState.service.phase === "starting"
                  ? "正在校验权重和加载模型，首次启动可能稍慢。"
                  : "选择模型与配置，启动后即可连接。"}
            </p>
          </div>
          <div class="service-actions">
            {#if active}<button
                class="danger-outline"
                disabled={busy || updating}
                onclick={() => perform(stop)}>停止并释放显存</button
              >{:else}<button
                class="primary"
                disabled={!initialized || busy || downloading || updating}
                onclick={() => perform(start)}>▶ 启动服务</button
              >{/if}
          </div>
        </div>
        <div class="connection-grid">
          <div>
            <label for="base-url">API Base URL</label>
            <div class="copy-field">
              <input id="base-url" value={baseUrl} readonly /><button
                disabled={!running}
                onclick={() => perform(() => copy(baseUrl))}>复制</button
              >
            </div>
            <div class="device-summary" aria-label="推理设备">
              {#if serviceGPU}
                <span
                  >{serviceGPU.name} · {gib(serviceGPU.total_memory_mib)} GiB</span
                >
                <span>驱动 {serviceGPU.driver_version}</span>
              {:else}<span
                  >{inventory.gpu_error?.message ??
                    "正在检查 NVIDIA 驱动…"}</span
                >{/if}
              <button
                class="text-button"
                disabled={busy}
                aria-label="重新检测显卡"
                title="重新检测显卡和显存预算"
                onclick={() =>
                  perform(async () => {
                    await refresh();
                    await estimate({ ...settings });
                  })}>↻</button
              >
            </div>
          </div>
          <div>
            <label for="model-id">模型名称</label>
            <div class="copy-field">
              <input id="model-id" value="hy-mt2" readonly /><button
                onclick={() => perform(() => copy("hy-mt2"))}>复制</button
              >
            </div>
          </div>
          <div>
            <label for="api-secret"
              >API Key<Help
                text="启用鉴权后，客户端需要 API Key；关闭后本机应用无需密钥即可调用。开关修改后重新部署生效。刷新密钥会生成新 Key 并重新部署，旧 Key 随即失效，请同步更新调用方。"
              /></label
            >
            <div class="copy-field">
              <input
                id="api-secret"
                value={running
                  ? currentAuth
                    ? "••••••••••••••••"
                    : "鉴权已关闭，无需 API Key"
                  : "启动服务后可复制"}
                readonly
              /><button
                disabled={!running || !currentAuth}
                onclick={() =>
                  perform(async () => copy(await invoke<string>("api_key")))}
                >复制</button
              >
            </div>
            <div class="key-actions">
              <button
                class="key-toggle"
                class:enabled={settings.apiKeyEnabled}
                role="switch"
                aria-label="API Key 鉴权"
                aria-checked={settings.apiKeyEnabled}
                title="切换 API Key 鉴权，重新部署后生效"
                onclick={() =>
                  (settings.apiKeyEnabled = !settings.apiKeyEnabled)}
              >
                <span class="mini-switch" aria-hidden="true"
                ></span>{settings.apiKeyEnabled ? "鉴权开启" : "鉴权关闭"}
              </button>
              <button
                class="key-refresh"
                title="生成新 Key 并重新部署，旧 Key 将失效"
                aria-label="生成新 API Key 并重新部署"
                disabled={busy ||
                  updating ||
                  !settings.apiKeyEnabled ||
                  !initialized ||
                  downloading ||
                  runtimeState.service.phase === "starting"}
                onclick={() => perform(() => restart(true))}>↻ 刷新密钥</button
              >
              {#if active && settings.apiKeyEnabled !== currentAuth}<span
                  class="key-pending">待应用</span
                >{/if}
            </div>
          </div>
        </div>
        <div class="service-foot">
          <span
            >接口 <code>/v1/chat/completions</code> · 支持流式响应 · 仅本机可访问</span
          >{#if runtimeState.service.port && runtimeState.service.port !== settings.port}<span
              class="warning-text"
              >端口占用，实际端口为 {runtimeState.service.port}</span
            >{/if}
        </div>
      </section>

      <div class="workspace-grid">
        <section class="panel configuration">
          <div class="section-heading">
            <h2>推理配置</h2>
            <div class="configuration-tools">
              <span>{dirty ? "有待应用的更改" : "修改后重新部署生效"}</span
              ><button
                class="reset-button"
                disabled={busy || updating}
                title="恢复 2K 上下文、Q8、30% 显存预算及自动模型/显卡/并发；保留 API 端口、鉴权和日志设置。"
                onclick={restoreDefaults}>↺ 恢复默认</button
              >
            </div>
          </div>
          <div class="form-grid">
            <label
              ><span
                >运行模型<Help
                  text="NVFP4 使用 RTX 50 系列的原生低精度计算，本项目优化后通常吞吐更高，仅支持已适配的 RTX 50（SM 120）构建。Q4_K_M 支持更多显卡。自动按显卡能力推荐，首次使用须先下载对应模型。"
                /></span
              >
              <select bind:value={settings.profile}
                ><option value="auto"
                  >自动选择{plan
                    ? ` · ${modelName(plan.profile)}`
                    : " · 检测中"}</option
                ><option value="fast">Hy-MT2-1.8B · NVFP4（RTX 50）</option
                ><option value="official">Hy-MT2-1.8B · Q4_K_M</option></select
              >
            </label>
            <label
              ><span
                >推理显卡<Help
                  text="自动优先选择可用显存最多且支持该模型的 NVIDIA 显卡。重新部署时计入本服务将释放的显存，不把其他应用的占用算作可用。"
                /></span
              >
              <select bind:value={settings.gpu}
                ><option value=""
                  >自动{plan
                    ? ` · ${plan.gpu.name.replace(/^NVIDIA\s+(GeForce\s+)?/, "")}`
                    : " · 检测中"}</option
                >{#each inventory.gpus as gpu}<option value={gpu.uuid}
                    >{gpu.name} · {gib(gpu.total_memory_mib)} GiB</option
                  >{/each}</select
              >
            </label>
            <label
              ><span
                >每请求上下文<Help
                  text="上下文是输入 + 输出的总 token 数，也包含提示词和对话模板。翻译时建议至少为输入 token 的 2 倍，并额外预留提示词空间；不同语言长度不同。token 不等同于字数。越长，KV 显存占用越大。"
                /></span
              >
              <select
                value={contextChoice}
                onchange={(event) => {
                  customContext = event.currentTarget.value === "custom";
                  if (!customContext)
                    settings.context = Number(event.currentTarget.value);
                }}
              >
                {#each contextOptions as value}<option value={String(value)}
                    >{value.toLocaleString()} tokens</option
                  >{/each}<option value="custom">自定义…</option>
              </select>
              {#if contextChoice === "custom"}<input
                  aria-label="自定义上下文 tokens"
                  type="number"
                  min="256"
                  max="32768"
                  step="1"
                  bind:value={settings.context}
                /><small>256–32,768 tokens / 请求</small>{/if}
            </label>
            <label
              ><span
                >请求并发<Help
                  text="同时处理请求的最大数量。请求充足时，提高并发通常提高总吞吐，但不会让每个请求都更快，收益受显卡限制。相同上下文下 KV 缓存随并发线性增长；模型权重不会成倍增加。0 按显存预算上限自动推荐，最多 128。"
                /></span
              >
              <input
                type="number"
                min="0"
                max="128"
                step="1"
                bind:value={settings.parallel}
              /><small
                >{settings.parallel === 0
                  ? `自动 → ${plan?.parallel ?? "计算中"} 并发`
                  : "最大 128 · KV 占用随并发增加"}</small
              >
            </label>
            <label class="wide"
              ><span>KV 缓存<Help text={performanceHelp} /></span>
              <select bind:value={settings.cache}>
                <option value="f16">F16 · 完整精度基准</option>
                <option value="q8_0"
                  >Q8 · 默认 · KV −46.9% · 吞吐损失参考 13.6%</option
                >
                <option value="q4_0">Q4 · KV −71.9% · 吞吐损失参考 14.6%</option
                >
              </select>
              <small>Q8：{savingLabel("q8_0")}；Q4：{savingLabel("q4_0")}</small
              >
              <small
                >相对同并发、同上下文的 F16。速度百分比为不同条件下的 RTX 5090
                历史批处理参考，非当前 API 性能预测。</small
              >
            </label>
            <label
              ><span
                >API 端口<Help
                  text="其他应用通过此本机端口访问 API。被占用时自动尝试后续 19 个端口，请以服务概览显示的实际 Base URL 为准。默认仅监听 127.0.0.1。"
                /></span
              >
              <input
                type="number"
                min="1024"
                max="65535"
                step="1"
                bind:value={settings.port}
              />
            </label>
            <label
              ><span
                >Micro batch（ubatch）<Help
                  text="每次实际提交 GPU 的 token 批次大小。较大时可能提高吞吐，但也会增加工作区显存。0 自动选择，显式值须不小于并发。通常保持自动即可。"
                /></span
              >
              <input
                type="number"
                min="0"
                max="8192"
                step="1"
                bind:value={settings.ubatch}
              /><small
                >{settings.ubatch === 0
                  ? `自动 → ${plan?.ubatch ?? "计算中"}`
                  : "数值须不小于并发"}</small
              >
            </label>
          </div>
          <div class="form-footer">
            <button
              disabled={busy || !initialized}
              onclick={() =>
                perform(async () => {
                  await save();
                  notify(
                    active
                      ? "配置已保存，点击重新部署后生效"
                      : "配置已保存，下次启动服务时生效",
                  );
                })}>保存配置</button
            >
            <button
              class="primary"
              disabled={busy ||
                updating ||
                downloading ||
                !initialized ||
                runtimeState.service.phase === "starting"}
              onclick={() => perform(() => (active ? restart() : start()))}
              >{active ? "重新部署 · 应用更改" : "应用配置并启动"}</button
            >
          </div>
        </section>
        <div class="right-stack">
          <section class="panel memory">
            <div class="section-heading">
              <h2>显存预算</h2>
              <span class="pill">估算值</span>
            </div>
            <div class="memory-limit">
              <label for="memory-percent"
                >显存预算上限<Help
                  text="默认用总显存的 30% 作为预算，包含模型、KV 缓存、工作区和安全余量。自动并发在这个预算与实际可用显存两者的较小值内推荐；手动并发也须通过预算检查。可调为 10%–100%。这是保守估算，不是驱动层的硬限额，实际占用随负载变化。"
                /></label
              >
              <div class="memory-limit-controls">
                <input
                  aria-label="显存预算比例滑块"
                  type="range"
                  min="10"
                  max="100"
                  step="1"
                  bind:value={settings.memoryPercent}
                /><input
                  id="memory-percent"
                  type="number"
                  min="10"
                  max="100"
                  step="1"
                  bind:value={settings.memoryPercent}
                /><span>%</span>
              </div>
              <small
                >{currentGPU && Number.isFinite(settings.memoryPercent)
                  ? `总显存的 ${settings.memoryPercent}% · 上限 ${gib((currentGPU.total_memory_mib * settings.memoryPercent) / 100)} GiB`
                  : "默认 30% · 自动选择合适的并发"}</small
              >
            </div>
            {#if currentGPU}<div class="availability">
                <div>
                  <span>当前空闲显存</span><strong
                    >{gib(currentGPU.free_memory_mib)} GiB</strong
                  >
                </div>
                <div>
                  <span
                    >重部署可用<Help
                      text="当前空闲 + 旧服务停止后释放的实际占用。预算中的安全余量只计算一次。驱动无法提供进程显存时，使用本服务启动前后空闲显存的差值估算，可能受其他程序影响；部署时会重新检查。"
                    /></span
                  ><strong>{gib(currentGPU.redeploy_available_mib)} GiB</strong>
                </div>
                {#if currentGPU.reclaimable_memory_mib > 0}<small
                    >含本服务可释放 {gib(currentGPU.reclaimable_memory_mib)} GiB{currentGPU.reclaim_is_estimate
                      ? "（估计）"
                      : ""}</small
                  >{/if}
              </div>{/if}
            {#if displayedPlan}<div class="memory-value">
                {gib(displayedPlan.budget.estimated_total_mib)}<span>GiB</span>
              </div>
              <p class="muted">
                {`${displayedPlan.parallel} 并发 · ${displayedPlan.context} tokens / 请求`}{dirty
                  ? " · 待应用"
                  : ""}
              </p>
              <div class="memory-bar" aria-hidden="true">
                {#each [{ size: displayedPlan.budget.model_reserve_mib, color: "#277b68" }, { size: displayedPlan.budget.kv_total_mib, color: "#8dc9ad" }, { size: displayedPlan.budget.workspace_reserve_mib, color: "#c2d0c7" }, { size: displayedPlan.budget.safety_margin_mib, color: "#e1e6df" }] as part}<span
                    style={`width:${(part.size / displayedPlan.budget.estimated_total_mib) * 100}%;background:${part.color}`}
                  ></span>{/each}
              </div>
              <dl>
                <div>
                  <dt><i style="background:#277b68"></i>模型权重预留</dt>
                  <dd>{gib(displayedPlan.budget.model_reserve_mib)} GiB</dd>
                </div>
                <div>
                  <dt><i style="background:#8dc9ad"></i>KV 缓存</dt>
                  <dd>{gib(displayedPlan.budget.kv_total_mib)} GiB</dd>
                </div>
                <div>
                  <dt><i style="background:#c2d0c7"></i>计算工作区</dt>
                  <dd>{gib(displayedPlan.budget.workspace_reserve_mib)} GiB</dd>
                </div>
                <div>
                  <dt><i style="background:#e1e6df"></i>安全余量</dt>
                  <dd>{gib(displayedPlan.budget.safety_margin_mib)} GiB</dd>
                </div>
              </dl>
              <div class="budget-note">
                基于显卡空闲显存和模型结构估算，实际占用随负载变化。量化 KV
                可能影响质量与速度。
              </div>{:else}<div class="empty-small">
                {estimating
                  ? "正在计算显存预算…"
                  : (planError?.message ?? "选择配置后显示显存预算。")}
              </div>
              {#if planError?.hint}<p class="muted">
                  {planError.hint}
                </p>{/if}{/if}
          </section>
          <p class="quiet-note">
            退出应用会停止推理并释放显存。最小化到托盘时，服务继续运行。
          </p>
        </div>
      </div>
    {:else if tab === "models"}
      <div class="section-heading">
        <h2>Hy-MT2-1.8B</h2>
        <button
          onclick={() =>
            perform(async () => {
              await invoke("open_folder", { kind: "models" });
            })}>打开模型目录 ↗</button
        >
      </div>
      <div class="model-grid">
        {#each inventory.models as model}<section class="panel model-card">
            <div class="model-top">
              <span class="model-icon"
                >{model.profile === "fast" ? "↯" : "▦"}</span
              ><span class="pill" class:installed={model.installed}
                >{model.installed ? "已下载" : "未下载"}</span
              >
            </div>
            <h2>{modelName(model.profile)}</h2>
            <p>
              {model.profile === "fast"
                ? "面向受支持的 RTX 50 架构，使用本项目优化的 NVFP4 权重。"
                : "腾讯 Q4_K_M 量化，适用于支持列表内的 GTX 16、RTX 20 / 30 / 40 / 50。"}
            </p>
            <div class="model-meta">
              <span>1.8B 参数</span><span>{fileSize(model.size_bytes)}</span
              ><span>GGUF</span>
            </div>
            {#if downloading && runtimeState.download.profile === model.profile}<div
                class="download-progress"
              >
                <progress
                  max={model.size_bytes}
                  value={runtimeState.progress[model.profile] ?? 0}
                ></progress><small
                  >{runtimeState.download.phase === "verifying"
                    ? "正在复制并校验…"
                    : `${fileSize(runtimeState.progress[model.profile] ?? 0)} / ${fileSize(model.size_bytes)} · 下载或校验中`}</small
                >
              </div>{:else if model.downloaded > 0 && !model.installed}<p
                class="muted"
              >
                已保留 {fileSize(model.downloaded)}，可继续下载。
              </p>{/if}
            <div class="model-buttons">
              <button
                class="primary"
                disabled={busy || active || downloading || updating}
                onclick={() => perform(() => download(model))}
                >{model.installed
                  ? "重新校验 / 修复"
                  : model.downloaded
                    ? "继续下载"
                    : "下载模型"}</button
              ><button
                disabled={busy || active || downloading}
                onclick={() => perform(() => importModel(model))}>导入</button
              >{#if model.installed || model.downloaded}<button
                  class="text-button"
                  disabled={busy || active || downloading}
                  onclick={() => perform(() => remove(model))}>删除</button
                >{/if}
            </div>
          </section>{/each}
      </div>
      {#if downloading}<div class="notice">
          <div>
            <strong>模型任务进行中</strong>
            <p>
              网络中断会自动重试，暂停后保留下载进度。SHA-256
              校验通过后才会启用模型。
            </p>
          </div>
          <button
            onclick={() =>
              perform(async () => {
                await invoke("task_stop", { kind: "download" });
                await poll();
                await refresh();
              })}>暂停任务</button
          >
        </div>{/if}
      {#if runtimeState.download.phase === "error"}<div
          class="notice danger"
          role="alert"
        >
          <div>
            <strong>{runtimeState.download.error?.message}</strong>
            <p>{runtimeState.download.error?.hint}</p>
          </div>
        </div>{/if}
      <div class="notice">
        <div>
          <strong>首次使用</strong>
          <p>
            RTX 50 可选 NVFP4，其他受支持显卡选择
            Q4_K_M。首次下载需要联网，后续推理可离线运行。导入仅接受此项目清单对应的
            fused.gguf 文件。
          </p>
          {#if active}<p>请先停止推理服务，再管理模型文件。</p>{/if}
        </div>
      </div>
      <p class="path-line">{runtimeState.portable ? "免安装版模型目录" : "共享模型目录"}：{inventory.model_dir}</p>
      <p class="muted">{runtimeState.portable ? "模型优先保存在程序旁的 models 文件夹，可随整个文件夹搬走。" : "安装版使用固定模型目录，升级无需重复下载。"}已有模型会校验后自动复用；未找到时可导入旧目录中的 GGUF 文件。</p>
      {#each inventory.model_warnings ?? [] as warning}
        <p class="muted">{warning}</p>
      {/each}
    {:else if tab === "settings"}
      <section class="panel settings-panel">
        <h2>启动与后台</h2>
        <div class="setting-row">
          <div>
            <strong
              >开机自启<Help
                text="通过 Tauri 官方自启插件注册当前程序。免安装版移动位置后，请关闭再开启此选项，以更新启动路径。静默自启和自动启动推理可分别配置。"
              /></strong
            >
            <p>登录 Windows 后启动 HyMT。</p>
          </div>
          <button
            role="switch"
            aria-label="开机自启"
            aria-checked={auto}
            class="switch"
            class:enabled={auto}
            disabled={busy}
            onclick={() => perform(toggleAuto)}><span></span></button
          >
        </div>
        {#each [{ key: "silentAutostart", title: "静默自启", detail: "开机自启时直接进入托盘；手动打开仍显示窗口。" }, { key: "startService", title: "打开应用时启动推理", detail: "使用已保存的配置。模型缺失或启动失败时显示诊断提示。" }, { key: "minimizeToTray", title: "最小化到托盘", detail: "最小化后隐藏任务栏窗口，服务保持运行。" }, { key: "closeToTray", title: "关闭窗口时保留在托盘", detail: "首次关闭窗口时询问并记住选择；可在这里修改。启用后从托盘「退出」释放显存。" }] as item}{@const key =
            item.key as keyof typeof prefs}
          <div class="setting-row">
            <div>
              <strong>{item.title}<Help text={item.detail} /></strong>
              <p>{item.detail}</p>
            </div>
            <button
              role="switch"
              aria-label={item.title}
              aria-checked={prefs[key]}
              class="switch"
              class:enabled={prefs[key]}
              disabled={busy}
              onclick={() =>
                perform(async () => {
                  const old = prefs[key];
                  prefs[key] = !old;
                  try {
                    await save();
                  } catch (e) {
                    prefs[key] = old;
                    throw e;
                  }
                })}><span></span></button
            >
          </div>{/each}
      </section>
      <section class="panel settings-panel">
        <h2>
          日志记录<Help
            text="默认仅在内存保留最近 512 行，每行最多 8,000 字符；退出即清空。开启文件日志后额外轮转保存 3 份、每份 2 MiB。完全关闭时不收集推理日志，但仍显示必要的服务状态和错误提示。模式修改后重新部署生效，已有历史日志不会自动删除。"
          />
        </h2>
        <div class="setting-row">
          <div>
            <strong>不写入日志文件</strong>
            <p>默认启用，只保留内存中的最近记录，退出即清空。</p>
          </div>
          <button
            role="switch"
            aria-label="不写入日志文件"
            aria-checked={settings.logMode !== "file"}
            class="switch"
            class:enabled={settings.logMode !== "file"}
            disabled={settings.logMode === "off"}
            onclick={() =>
              (settings.logMode =
                settings.logMode === "file" ? "memory" : "file")}
            ><span></span></button
          >
        </div>
        <div class="setting-row">
          <div>
            <strong>完全关闭日志</strong>
            <p>默认关闭。开启后不记录推理日志，仅显示状态和错误提示。</p>
          </div>
          <button
            role="switch"
            aria-label="完全关闭日志"
            aria-checked={settings.logMode === "off"}
            class="switch"
            class:enabled={settings.logMode === "off"}
            onclick={() =>
              (settings.logMode =
                settings.logMode === "off" ? "memory" : "off")}
            ><span></span></button
          >
        </div>
        <div class="form-footer">
          <button
            disabled={busy}
            onclick={() =>
              perform(async () => {
                await save();
                notify(
                  active
                    ? "已保存，重新部署后生效"
                    : "已保存，下次启动服务生效",
                );
              })}>保存配置</button
          >
        </div>
      </section>
      <section class="panel update-panel">
        <div class="section-heading">
          <div>
            <h2>应用更新</h2>
            <p>
              当前版本 {version} · {runtimeState.portable
                ? "免安装版"
                : "安装版"}
            </p>
          </div>
          <button
            disabled={busy || updating}
            onclick={() => perform(checkUpdate)}>检查更新</button
          >
        </div>
        {#if update}<div class="notice">
            <div>
              <strong>发现新版本 {update.version}</strong>
              <p class="release-notes">
                {update.body ?? "包含功能改进与修复。"}
              </p>
            </div>
            <button
              class="primary"
              disabled={updating || busy}
              onclick={() => perform(installUpdate)}>下载并更新</button
            >
          </div>{/if}{#if updating}<p>
            正在下载并验证更新：{fileSize(updateBytes)}{updateTotal
              ? ` / ${fileSize(updateTotal)}`
              : ""}
          </p>{/if}
        <p class="muted">
          更新包经过签名验证；安装时停止后台服务。模型和个人配置独立保存。
        </p>
      </section>
      <section class="panel">
        <div class="section-heading">
          <div>
            <h2>关于 HyMT</h2>
            <p>基于 Tauri 2、Svelte、Hy-MT2 与 llama.cpp。</p>
          </div>
          <button
            onclick={() =>
              perform(async () => {
                await invoke("open_folder", { kind: "licenses" });
              })}>第三方许可 ↗</button
          >
        </div>
        <p class="path-line">数据目录：{inventory.data_dir}</p>
      </section>
    {:else}
      <section class="panel">
        <div class="section-heading">
          <div>
            <h2>推理服务日志</h2>
            <p>
              {logMode === "off"
                ? "日志已关闭，仅保留必要的状态和错误提示。"
                : logMode === "file"
                  ? "内存保留最近 512 行；文件最多 3 份 × 2 MiB。"
                  : "仅保留内存中的最近 512 行，退出即清空，不写日志文件。"}
            </p>
          </div>
          <div class="button-row">
            <button
              onclick={() =>
                perform(async () => (logs = await rpc<string>("logs")))}
              >刷新</button
            ><button onclick={() => perform(() => copy(logs))}>复制日志</button
            ><button
              onclick={() =>
                perform(async () => {
                  await invoke("open_folder", { kind: "logs" });
                })}>打开目录 ↗</button
            >
          </div>
        </div>
        <pre class="logs">{logs ||
            "尚无推理日志。启动一次服务后，运行信息会显示在这里。"}</pre>
      </section>
      <div class="diagnostic-grid">
        {#each [{ title: "显存不足", detail: "降低并发或上下文；若预算比例过低，可调高显存预算上限。关闭其他占用 GPU 的程序后重新检测。" }, { title: "连接不上 API", detail: "确认服务状态为运行中，复制界面显示的实际 Base URL 和 API Key。客户端模型名填写 hy-mt2。" }, { title: "模型 / 运行库损坏", detail: "模型页可重新校验和下载。缺失 DLL 时重新解压完整运行包，或重新安装应用。" }] as tip}<section
            class="panel"
          >
            <h3>{tip.title}</h3>
            <p>{tip.detail}</p>
          </section>{/each}
      </div>
    {/if}
    <footer>
      <span><span class="dot online"></span> 本地推理 · OpenAI 兼容接口</span
      ><button
        class="text-button"
        disabled={updating}
        onclick={() =>
          perform(async () => {
            await invoke("quit");
          })}>退出应用并停止服务</button
      >
    </footer>
  </main>
</div>
{#if toast}<div class="toast" role="status">✓ {toast}</div>{/if}
