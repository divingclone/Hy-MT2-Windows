#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod job;
mod webview;

use serde_json::{json, Value};
use std::{
    collections::VecDeque,
    fs,
    io::{BufRead, BufReader, Write},
    os::windows::process::CommandExt,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager,
};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogResult};
use tauri_plugin_opener::OpenerExt;
use tauri_plugin_store::StoreExt;
use tauri_plugin_updater::UpdaterExt;

static SEQUENCE: AtomicU64 = AtomicU64::new(0);
const CREATE_NO_WINDOW: u32 = 0x08000000;

struct Worker {
    child: Child,
    job: job::Job,
    output: PathBuf,
    readers: Vec<std::thread::JoinHandle<()>>,
}
impl Worker {
    fn stop(&mut self) {
        self.job.terminate();
        let _ = self.child.wait();
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
    }
    fn read(&self) -> Value {
        fs::read(&self.output)
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_else(|| json!({"phase":"starting"}))
    }
    fn status(&mut self) -> Value {
        let mut value = self.read();
        if let Some(result) = value.get("result").cloned() {
            value = result;
        }
        if let Ok(Some(code)) = self.child.try_wait() {
            // A worker killed from outside cannot leave its inference descendant
            // running merely because the desktop still holds the Job handle.
            self.job.terminate();
            let phase = value.get("phase").and_then(Value::as_str).unwrap_or("");
            if phase != "error"
                && (!code.success()
                    || matches!(phase, "starting" | "ready" | "downloading" | "verifying"))
            {
                return json!({"phase":"error", "error":{"message":format!("后台进程异常退出（{code}）。"), "hint":"推理子进程已停止。请打开运行诊断，检查完整运行库和目录权限。"}});
            }
        }
        value
    }
    fn alive(&mut self) -> bool {
        self.child
            .try_wait()
            .map(|value| value.is_none())
            .unwrap_or(false)
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.stop();
        let _ = fs::remove_file(&self.output);
    }
}

struct Runtime {
    root: PathBuf,
    data: PathBuf,
    models: PathBuf,
    model_registry: Option<PathBuf>,
    portable: bool,
    service: Mutex<Option<Worker>>,
    download: Mutex<Option<Worker>>,
    pending_update: Mutex<Option<tauri_plugin_updater::Update>>,
    updating: AtomicBool,
    deploying: AtomicBool,
    close_dialog: AtomicBool,
    logs: Arc<Mutex<VecDeque<String>>>,
    logging_enabled: AtomicBool,
}
impl Runtime {
    fn spawn(&self, action: &str, args: Value) -> Result<Worker, String> {
        self.spawn_context(action, args, Value::Null)
    }
    fn spawn_context(&self, action: &str, args: Value, service: Value) -> Result<Worker, String> {
        let python = [
            self.root.join("runtime/python/python.exe"),
            self.root
                .join("runtime/python/cpython-3.12-windows-x86_64-none/python.exe"),
        ]
        .into_iter()
        .find(|path| path.is_file())
        .ok_or("缺少随包 Python，请使用完整安装包或免安装包。")?;
        let id = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let output = self
            .data
            .join(format!("tasks/{}-{id}.json", std::process::id()));
        let mut child = Command::new(python)
            .args(["-E", "-s", "-B", "-u", "-X", "utf8"])
            .arg(self.root.join("scripts/desktop_bridge.py"))
            .current_dir(&self.data)
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("无法启动后台组件：{e}"))?;
        let job = job::Job::attach(&mut child)
            .map_err(|e| format!("无法绑定后台进程，已取消启动：{e}"))?;
        let capture = action == "service" && args["logMode"].as_str() != Some("off");
        if action == "service" {
            self.logs.lock().unwrap().clear();
            self.logging_enabled.store(capture, Ordering::SeqCst);
        }
        let mut readers = Vec::new();
        let streams: Vec<Box<dyn std::io::Read + Send>> = vec![
            Box::new(child.stdout.take().unwrap()),
            Box::new(child.stderr.take().unwrap()),
        ];
        for stream in streams {
            let logs = self.logs.clone();
            readers.push(std::thread::spawn(move || {
                for line in BufReader::new(stream).lines().map_while(Result::ok) {
                    if capture {
                        let mut buffer = logs.lock().unwrap();
                        buffer.push_back(line.chars().take(8000).collect());
                        while buffer.len() > 512 {
                            buffer.pop_front();
                        }
                    }
                }
            }));
        }
        let mut worker = Worker {
            child,
            job,
            output,
            readers,
        };
        let executable_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()));
        let request = json!({"root":self.root,"data":self.data,"model_dir":self.models,
            "model_registry":self.model_registry,
            "model_search_roots":[self.data,self.root,executable_dir,self.model_registry.as_ref().and_then(|p| p.parent())],
            "output":worker.output,"action":action,"args":args,"service":service});
        let mut stdin = worker.child.stdin.take().ok_or("无法建立后台控制通道")?;
        writeln!(stdin, "{request}").map_err(|e| e.to_string())?;
        Ok(worker)
    }
    fn stop(&self) {
        self.service.lock().unwrap().take();
        self.download.lock().unwrap().take();
        let _ = fs::remove_file(self.data.join("service-private.json"));
    }
    fn deploy(&self, args: Value) -> Result<(), String> {
        if self.updating.load(Ordering::SeqCst) || self.deploying.swap(true, Ordering::SeqCst) {
            return Err("更新或部署正在进行中。".into());
        }
        let result = (|| {
            if self
                .download
                .lock()
                .unwrap()
                .as_mut()
                .is_some_and(Worker::alive)
            {
                return Err("请等待模型任务完成。".into());
            }
            let snapshot = self
                .service
                .lock()
                .unwrap()
                .as_mut()
                .map(Worker::status)
                .unwrap_or(Value::Null);
            let mut check = self.spawn_context("validate_deploy", args.clone(), snapshot)?;
            let deadline = Instant::now() + Duration::from_secs(600);
            while check.alive() {
                if Instant::now() > deadline {
                    return Err("配置检查超时，原服务继续运行。".into());
                }
                std::thread::sleep(Duration::from_millis(60));
            }
            unwrap_result(check.read())?;
            let mut slot = self.service.lock().unwrap();
            slot.take(); // Drop closes the Job and waits for all descendants before replacement.
            let _ = fs::remove_file(self.data.join("service-private.json"));
            *slot = Some(self.spawn("service", args)?);
            Ok(())
        })();
        self.deploying.store(false, Ordering::SeqCst);
        result
    }
}

#[tauri::command]
async fn redeploy(app: tauri::AppHandle, args: Value) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || app.state::<Runtime>().deploy(args))
        .await
        .map_err(|e| e.to_string())?
}

fn unwrap_result(value: Value) -> Result<Value, String> {
    if value.get("ok").and_then(Value::as_bool) == Some(true) {
        Ok(value["result"].clone())
    } else {
        Err(value.get("error").unwrap_or(&value).to_string())
    }
}

#[tauri::command]
async fn rpc(app: tauri::AppHandle, action: String, args: Option<Value>) -> Result<Value, String> {
    if !["inventory", "estimate", "logs", "remove_model"].contains(&action.as_str()) {
        return Err("不允许的后台操作".into());
    }
    tauri::async_runtime::spawn_blocking(move || {
        let runtime = app.state::<Runtime>();
        if action == "logs" {
            if !runtime.logging_enabled.load(Ordering::SeqCst) {
                return Ok(json!("日志已完全关闭，不记录运行日志。"));
            }
            return Ok(json!(runtime
                .logs
                .lock()
                .unwrap()
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")));
        }
        if runtime.updating.load(Ordering::SeqCst) {
            return Err("更新期间请稍候。".into());
        }
        // Prevent removal from racing with either startup/hash verification or download.
        let mut service_guard = runtime.service.lock().unwrap();
        let mut download_guard = runtime.download.lock().unwrap();
        let service_snapshot = service_guard
            .as_mut()
            .map(Worker::status)
            .unwrap_or(Value::Null);
        if action == "remove_model"
            && (runtime.deploying.load(Ordering::SeqCst)
                || service_guard.as_mut().is_some_and(Worker::alive)
                || download_guard.as_mut().is_some_and(Worker::alive))
        {
            return Err("请先停止服务和模型任务，再删除模型。".into());
        }
        let keep_guards = action == "remove_model";
        let service_hold = if keep_guards {
            Some(service_guard)
        } else {
            drop(service_guard);
            None
        };
        let download_hold = if keep_guards {
            Some(download_guard)
        } else {
            drop(download_guard);
            None
        };
        let mut worker =
            runtime.spawn_context(&action, args.unwrap_or(json!({})), service_snapshot)?;
        // Initial model reuse may hash/copy gigabytes from an older portable
        // directory, particularly across disks. Keep the UI thread responsive.
        let deadline =
            Instant::now() + Duration::from_secs(if action == "inventory" { 180 } else { 35 });
        while worker.alive() {
            if Instant::now() > deadline {
                return Err("操作超时，后台任务已取消，请检查驱动或服务状态。".into());
            }
            std::thread::sleep(Duration::from_millis(60));
        }
        drop(service_hold);
        drop(download_hold);
        unwrap_result(worker.read())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn task_start(app: tauri::AppHandle, kind: String, args: Value) -> Result<(), String> {
    let runtime = app.state::<Runtime>();
    if runtime.updating.load(Ordering::SeqCst) || runtime.deploying.load(Ordering::SeqCst) {
        return Err("更新期间请稍候。".into());
    }
    let mut service = runtime.service.lock().unwrap();
    let mut download = runtime.download.lock().unwrap();
    if !["service", "download", "import_model"].contains(&kind.as_str()) {
        return Err("未知任务".into());
    }
    if kind != "service" && service.as_mut().is_some_and(Worker::alive) {
        return Err("请先停止推理服务，再下载或导入模型。".into());
    }
    if kind == "service" && download.as_mut().is_some_and(Worker::alive) {
        return Err("请等待模型任务完成。".into());
    }
    let slot = if kind == "service" {
        &mut *service
    } else {
        &mut *download
    };
    if slot.as_mut().is_some_and(Worker::alive) {
        return Err("任务正在运行。".into());
    }
    *slot = Some(runtime.spawn(&kind, args)?);
    Ok(())
}

#[tauri::command]
fn task_stop(app: tauri::AppHandle, kind: String) -> Result<(), String> {
    let runtime = app.state::<Runtime>();
    if runtime.deploying.load(Ordering::SeqCst) {
        return Err("正在重新部署，请稍候。".into());
    }
    match kind.as_str() {
        "service" => {
            runtime.service.lock().unwrap().take();
            let _ = fs::remove_file(runtime.data.join("service-private.json"));
        }
        "download" => {
            runtime.download.lock().unwrap().take();
        }
        _ => return Err("未知任务".into()),
    }
    Ok(())
}

#[tauri::command]
fn status(app: tauri::AppHandle) -> Value {
    let runtime = app.state::<Runtime>();
    let service = runtime
        .service
        .lock()
        .unwrap()
        .as_mut()
        .map(Worker::status)
        .unwrap_or(json!({"phase":"stopped"}));
    let download = runtime
        .download
        .lock()
        .unwrap()
        .as_mut()
        .map(Worker::status)
        .unwrap_or(json!({"phase":"idle"}));
    let progress: Value = fs::read(runtime.root.join("models/manifest.json"))
        .ok()
        .and_then(|v| serde_json::from_slice::<Value>(&v).ok())
        .map(|manifest| {
            let mut values = serde_json::Map::new();
            if let Some(files) = manifest["files"].as_object() {
                for (profile, item) in files {
                    let spec = if profile == "fast" {
                        &manifest
                    } else {
                        &manifest["checkpoints"][profile]
                    };
                    if let (Some(directory), Some(checkpoint_files)) = (
                        spec["checkpoint_dir"].as_str(),
                        spec["checkpoint_files"].as_object(),
                    ) {
                        let base = runtime.models.join(item["sha256"].as_str().unwrap_or(""));
                        let staging = base.join(format!("{directory}.download"));
                        let completed = base.join(directory);
                        let mut bytes = 0u64;
                        for (name, metadata) in checkpoint_files {
                            let size = metadata["size_bytes"].as_u64().unwrap_or(0);
                            let received = fs::metadata(completed.join(name))
                                .or_else(|_| fs::metadata(staging.join(name)))
                                .or_else(|_| fs::metadata(staging.join(format!("{name}.part"))))
                                .map(|s| s.len())
                                .unwrap_or(0);
                            bytes += received.min(size);
                        }
                        values.insert(profile.clone(), json!(bytes));
                    }
                }
            }
            Value::Object(values)
        })
        .unwrap_or(json!({}));
    json!({"service": service, "download": download, "progress": progress, "portable":runtime.portable})
}

#[tauri::command]
fn preferences(app: tauri::AppHandle, value: Option<Value>) -> Result<Value, String> {
    let store = app
        .store(app.state::<Runtime>().data.join("preferences.json"))
        .map_err(|e| e.to_string())?;
    if let Some(value) = value {
        if store
            .get("preferences")
            .is_some_and(|old| old["closeToTray"] != value["closeToTray"])
        {
            store.set("closeChoiceMade", true);
        }
        store.set("preferences", value);
        store.save().map_err(|e| e.to_string())?;
    }
    Ok(store.get("preferences").unwrap_or(json!({})))
}

#[tauri::command]
fn open_folder(app: tauri::AppHandle, kind: String) -> Result<(), String> {
    let runtime = app.state::<Runtime>();
    let path = match kind.as_str() {
        "models" => runtime.models.clone(),
        "logs" => runtime.data.join("logs"),
        "licenses" => runtime.root.join("licenses"),
        _ => return Err("未知目录".into()),
    };
    app.opener()
        .open_path(path.to_string_lossy(), None::<&str>)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn api_key(app: tauri::AppHandle) -> Result<String, String> {
    let runtime = app.state::<Runtime>();
    if runtime
        .service
        .lock()
        .unwrap()
        .as_mut()
        .map(Worker::status)
        .and_then(|s| s["phase"].as_str().map(String::from))
        != Some("ready".into())
    {
        return Err("服务尚未就绪。".into());
    }
    let private: Value = serde_json::from_slice(
        &fs::read(runtime.data.join("service-private.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(private["key"].as_str().unwrap_or_default().into())
}

#[tauri::command]
fn quit(app: tauri::AppHandle) {
    app.state::<Runtime>().stop();
    app.exit(0);
}

#[tauri::command]
async fn check_update(app: tauri::AppHandle) -> Result<Value, String> {
    let runtime = app.state::<Runtime>();
    if app
        .config()
        .plugins
        .0
        .get("updater")
        .and_then(|v| v["pubkey"].as_str())
        .unwrap_or_default()
        .is_empty()
    {
        return Err("此开发构建尚未配置更新公钥和发布地址。正式发布需先运行签名打包流程。".into());
    }
    let mut builder = app.updater_builder();
    if runtime.portable {
        builder = builder.target("windows-x86_64-portable");
    }
    let update = builder
        .build()
        .map_err(|e| e.to_string())?
        .check()
        .await
        .map_err(|e| format!("检查更新失败：{e}"))?;
    let result = update
        .as_ref()
        .map(|u| json!({"version":u.version,"body":u.body}))
        .unwrap_or(Value::Null);
    *runtime.pending_update.lock().unwrap() = update;
    Ok(result)
}

#[tauri::command]
async fn install_update(app: tauri::AppHandle) -> Result<(), String> {
    let runtime = app.state::<Runtime>();
    if runtime.deploying.load(Ordering::SeqCst) {
        return Err("正在重新部署，请稍候。".into());
    }
    if runtime.updating.swap(true, Ordering::SeqCst) {
        return Err("更新已在进行中。".into());
    }
    let result = async {
        let update = runtime
            .pending_update
            .lock()
            .unwrap()
            .take()
            .ok_or("请先检查更新。")?;
        let bytes = update
            .download(
                |chunk, total| {
                    let _ = app.emit("update-progress", json!({"chunk":chunk,"total":total}));
                },
                || {},
            )
            .await
            .map_err(|e| e.to_string())?;
        runtime.stop();
        if runtime.portable {
            let staging = runtime.data.join("updates");
            fs::create_dir_all(&staging).map_err(|e| e.to_string())?;
            let archive = staging.join("verified-update.zip");
            fs::write(&archive, bytes).map_err(|e| e.to_string())?;
            // Copy the helper/runtime outside the installation before replacing it.
            // Preparation validates archive paths and stages all bytes before exit.
            let python = runtime.root.join("runtime/python/python.exe");
            let exe = std::env::current_exe().map_err(|e| e.to_string())?;
            let helper = runtime.root.join("scripts/desktop_update.py");
            let output = Command::new(&python)
                .arg(&helper)
                .arg("prepare")
                .arg(&archive)
                .arg(exe.parent().unwrap())
                .arg(&staging)
                .creation_flags(CREATE_NO_WINDOW)
                .output()
                .map_err(|e| e.to_string())?;
            if !output.status.success() {
                return Err(String::from_utf8_lossy(&output.stderr).to_string());
            }
            Command::new(staging.join("helper-python/python.exe"))
                .arg(staging.join("desktop_update.py"))
                .arg("apply")
                .arg(exe.parent().unwrap())
                .arg(&staging)
                .arg(std::process::id().to_string())
                .creation_flags(CREATE_NO_WINDOW)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|e| e.to_string())?;
            app.exit(0);
        } else {
            update.install(bytes).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    .await;
    runtime.updating.store(false, Ordering::SeqCst);
    result
}

fn show(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

fn main() {
    match webview::prepare() {
        Ok(true) => (),
        Ok(false) => return,
        Err(error) => {
            webview::show_error(&error);
            return;
        }
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _, _| show(app)))
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            Some(vec!["--autostart"]),
        ))
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(
            tauri_plugin_log::Builder::new()
                .clear_targets()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            let exe = std::env::current_exe()?;
            let portable = exe.parent().unwrap().join("portable.json").is_file();
            let root = if cfg!(debug_assertions) {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .canonicalize()?
            } else {
                app.path().resource_dir()?.join("payload")
            };
            let data = if portable {
                exe.parent().unwrap().join("data")
            } else {
                app.path().app_local_data_dir()?
            };
            fs::create_dir_all(data.join("tasks"))?;
            // Prefer the user's explicit LOCALAPPDATA path: launching from an
            // MSIX-hosted development tool can redirect the Known Folder API.
            let shared_data = std::env::var_os("LOCALAPPDATA")
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .map(|path| path.join(&app.config().identifier))
                .unwrap_or(app.path().app_local_data_dir()?);
            let models = if portable {
                exe.parent().unwrap().join("models")
            } else {
                shared_data.join("models")
            };
            let model_registry = Some(shared_data.join("model-locations.json"));
            fs::create_dir_all(&models)?;
            fs::create_dir_all(data.join("logs"))?;
            app.manage(Runtime {
                root,
                data,
                models,
                model_registry,
                portable,
                service: Mutex::new(None),
                download: Mutex::new(None),
                pending_update: Mutex::new(None),
                updating: AtomicBool::new(false),
                deploying: AtomicBool::new(false),
                close_dialog: AtomicBool::new(false),
                logs: Arc::new(Mutex::new(VecDeque::new())),
                logging_enabled: AtomicBool::new(true),
            });
            let open = MenuItem::with_id(app, "show", "打开 HyMT", true, None::<&str>)?;
            let stop = MenuItem::with_id(app, "stop", "停止推理并释放显存", true, None::<&str>)?;
            let exit =
                MenuItem::with_id(app, "quit", "退出 HyMT（停止所有任务）", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open, &stop, &exit])?;
            TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("HyMT · 本地翻译")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => show(app),
                    "stop" => {
                        let _ = task_stop(app.clone(), "service".into());
                    }
                    "quit" => quit(app.clone()),
                    _ => (),
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show(tray.app_handle());
                    }
                })
                .build(app)?;
            let prefs = preferences(app.handle().clone(), None).unwrap_or(json!({}));
            let autostart = std::env::args().any(|arg| arg == "--autostart");
            if !(autostart && prefs["silentAutostart"].as_bool().unwrap_or(false)) {
                show(app.handle());
            }
            if prefs["startService"].as_bool().unwrap_or(false) {
                if let Err(error) = task_start(
                    app.handle().clone(),
                    "service".into(),
                    prefs.get("settings").cloned().unwrap_or(json!({})),
                ) {
                    log::error!("Automatic service startup: {error}");
                    show(app.handle());
                }
            }
            // A silent startup error must not remain invisible in the tray.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if autostart {
                    for _ in 0..100 {
                        tokio::time::sleep(Duration::from_secs(2)).await;
                        let state = status(handle.clone());
                        if state["service"]["phase"] == "error" {
                            show(&handle);
                            break;
                        }
                        if state["service"]["phase"] != "starting" {
                            break;
                        }
                    }
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            let prefs = preferences(window.app_handle().clone(), None).unwrap_or(json!({}));
            match event {
                tauri::WindowEvent::CloseRequested { api, .. } => {
                    let app = window.app_handle();
                    let runtime = app.state::<Runtime>();
                    let store = app.store(runtime.data.join("preferences.json"));
                    let choice_made = store.as_ref().ok().and_then(|s| s.get("closeChoiceMade")).and_then(|v| v.as_bool()).unwrap_or(false);
                    if !choice_made {
                        api.prevent_close();
                        if runtime.close_dialog.swap(true, Ordering::SeqCst) { return; }
                        let handle = app.clone();
                        app.dialog().message("退出程序会停止推理并释放显存；最小化到托盘后 API 继续运行。选择将被记住，可在应用设置中修改。")
                            .title("关闭 HyMT")
                            .parent(window)
                            .buttons(MessageDialogButtons::YesNoCancelCustom("退出程序".into(), "最小化到托盘".into(), "取消".into()))
                            .show_with_result(move |result| {
                                handle.state::<Runtime>().close_dialog.store(false, Ordering::SeqCst);
                                let tray = match result {
                                    MessageDialogResult::Custom(ref label) if label == "最小化到托盘" => true,
                                    MessageDialogResult::Custom(ref label) if label == "退出程序" => false,
                                    _ => return,
                                };
                                if let Ok(store) = handle.store(handle.state::<Runtime>().data.join("preferences.json")) {
                                    let mut current = store.get("preferences").unwrap_or(json!({}));
                                    current["closeToTray"] = json!(tray);
                                    store.set("preferences", current.clone());
                                    store.set("closeChoiceMade", true);
                                    if store.save().is_err() { return; }
                                    let _ = handle.emit("preferences-changed", current);
                                }
                                if tray {
                                    if let Some(window) = handle.get_webview_window("main") { let _ = window.hide(); }
                                } else { quit(handle); }
                            });
                        return;
                    }
                    if prefs["closeToTray"].as_bool().unwrap_or(false) {
                        api.prevent_close();
                        let _ = window.hide();
                    } else {
                        window.app_handle().state::<Runtime>().stop();
                    }
                }
                tauri::WindowEvent::Resized(_)
                    if prefs["minimizeToTray"].as_bool().unwrap_or(true) =>
                {
                    if window.is_minimized().unwrap_or(false) {
                        let _ = window.hide();
                    }
                }
                _ => (),
            }
        })
        .invoke_handler(tauri::generate_handler![
            rpc,
            task_start,
            task_stop,
            redeploy,
            status,
            preferences,
            open_folder,
            api_key,
            quit,
            check_update,
            install_update
        ])
        .build(tauri::generate_context!())
        .expect("HyMT 初始化失败")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                app.state::<Runtime>().stop();
            }
        });
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::{io::Read, net::TcpStream};

    #[test]
    #[ignore = "requires bundled runtime, verified model and NVIDIA GPU; run explicitly"]
    fn real_gpu_api_and_stop() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        let data = root.join(format!(".local/desktop-gpu-test-{}", std::process::id()));
        fs::create_dir_all(data.join("models")).unwrap();
        fs::create_dir_all(data.join("tasks")).unwrap();
        let name = "Hy-MT2-1.8B-NVFP4-vllm.zip";
        if !data.join("models").join(name).exists() {
            fs::hard_link(
                root.join("models").join(name),
                data.join("models").join(name),
            )
            .unwrap();
        }
        let runtime = Runtime {
            models: data.join("shared-models"),
            model_registry: None,
            root,
            data,
            portable: false,
            service: Mutex::new(None),
            download: Mutex::new(None),
            pending_update: Mutex::new(None),
            updating: AtomicBool::new(false),
            deploying: AtomicBool::new(false),
            close_dialog: AtomicBool::new(false),
            logs: Arc::new(Mutex::new(VecDeque::new())),
            logging_enabled: AtomicBool::new(true),
        };
        let mut worker = runtime.spawn("service", json!({"profile":"fast", "parallel":1, "context":1024,"cache":"int8_per_token_head","port":19876})).unwrap();
        let deadline = Instant::now() + Duration::from_secs(600);
        let mut next_log = Instant::now() + Duration::from_secs(30);
        let status = loop {
            let status = worker.status();
            assert_ne!(status["phase"], "error", "{status}");
            if status["phase"] == "ready" {
                break status;
            }
            if Instant::now() >= next_log {
                for line in runtime
                    .logs
                    .lock()
                    .unwrap()
                    .iter()
                    .rev()
                    .take(12)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                {
                    eprintln!("{line}");
                }
                next_log = Instant::now() + Duration::from_secs(30);
            }
            assert!(
                Instant::now() < deadline,
                "GPU startup timed out: {status}\n{:?}",
                runtime.logs.lock().unwrap()
            );
            std::thread::sleep(Duration::from_millis(250));
        };
        let private: Value =
            serde_json::from_slice(&fs::read(runtime.data.join("service-private.json")).unwrap())
                .unwrap();
        let key = private["key"].as_str().unwrap();
        let port = status["port"].as_u64().unwrap() as u16;
        fn request(port: u16, key: &str, method: &str, path: &str, body: &str) -> String {
            let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(45)))
                .unwrap();
            write!(stream, "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {key}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            let mut result = String::new();
            stream.read_to_string(&mut result).unwrap();
            result
        }
        let models = request(port, key, "GET", "/v1/models", "");
        assert!(models.contains("200 OK") && models.contains("hy-mt2"));
        let unauthorized = request(port, "wrong", "GET", "/v1/models", "");
        assert!(unauthorized.contains("401 Unauthorized"));
        let completion = request(
            port,
            key,
            "POST",
            "/v1/chat/completions",
            r#"{"model":"hy-mt2","messages":[{"role":"user","content":"Translate into Chinese, only output translation: Hello world."}],"max_tokens":64,"stream":false}"#,
        );
        assert!(
            completion.contains("200 OK") && completion.contains("\"content\""),
            "{completion}"
        );
        let stream = request(
            port,
            key,
            "POST",
            "/v1/chat/completions",
            r#"{"model":"hy-mt2","messages":[{"role":"user","content":"Translate into Chinese: Thank you."}],"max_tokens":64,"stream":true}"#,
        );
        assert!(
            stream.contains("data: ") && stream.contains("[DONE]"),
            "{stream}"
        );
        let web = request(port, key, "GET", "/", "");
        assert!(!web.contains("<html"), "Web UI must be disabled");
        worker.stop();
        for _ in 0..40 {
            if TcpStream::connect(("127.0.0.1", port)).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(
            TcpStream::connect(("127.0.0.1", port)).is_err(),
            "API listener survived stop"
        );
        assert!(
            runtime.data.join("api-key.txt").is_file(),
            "API key must persist across service restart"
        );
        assert!(
            !runtime.logs.lock().unwrap().is_empty(),
            "default logs must be captured in RAM"
        );
        assert!(
            !runtime.data.join("logs/inference.log").exists(),
            "default logs must not touch disk"
        );
        *runtime.service.lock().unwrap() = Some(worker);
        fn wait_ready(runtime: &Runtime) -> Value {
            let deadline = Instant::now() + Duration::from_secs(600);
            loop {
                let status = runtime.service.lock().unwrap().as_mut().unwrap().status();
                assert_ne!(status["phase"], "error", "{status}");
                if status["phase"] == "ready" {
                    return status;
                }
                assert!(Instant::now() < deadline, "{status}");
                std::thread::sleep(Duration::from_millis(250));
            }
        }
        let config = json!({"profile":"fast", "parallel":1, "context":1537, "cache":"int8_per_token_head", "port":19876, "apiKeyEnabled":false, "logMode":"off"});
        runtime.deploy(config.clone()).unwrap();
        let next = wait_ready(&runtime);
        assert_eq!(next["plan"]["context"], 1537);
        assert_eq!(next["plan"]["settings"]["apiKeyEnabled"], false);
        assert!(
            request(port, "", "GET", "/v1/models", "").contains("200 OK"),
            "disabled auth should accept requests without a key"
        );
        assert!(
            runtime.logs.lock().unwrap().is_empty(),
            "off mode must not capture logs"
        );
        assert!(!runtime.data.join("logs/inference.log").exists());
        let pid = next["pid"].clone();
        let mut invalid = config.clone();
        invalid["context"] = json!(0);
        assert!(runtime.deploy(invalid).is_err());
        assert_eq!(
            wait_ready(&runtime)["pid"],
            pid,
            "invalid deployment must preserve the original service"
        );
        assert!(request(port, "", "GET", "/v1/models", "").contains("200 OK"));
        let mut rotated = config;
        rotated["apiKeyEnabled"] = json!(true);
        rotated["_rotate_key"] = json!(true);
        rotated["logMode"] = json!("file");
        runtime.deploy(rotated).unwrap();
        assert_ne!(wait_ready(&runtime)["pid"], pid);
        let new_key = fs::read_to_string(runtime.data.join("api-key.txt")).unwrap();
        assert_ne!(new_key, key);
        assert!(request(port, key, "GET", "/v1/models", "").contains("401 Unauthorized"));
        assert!(request(port, &new_key, "GET", "/v1/models", "").contains("200 OK"));
        assert!(runtime.data.join("logs/inference.log").is_file());
        assert!(!fs::read_to_string(runtime.data.join("logs/inference.log"))
            .unwrap()
            .contains(&new_key));
        runtime.stop();
        // TCP teardown can trail the signalled process exit briefly on Windows.
        for _ in 0..40 {
            if TcpStream::connect(("127.0.0.1", port)).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(TcpStream::connect(("127.0.0.1", port)).is_err());
        assert!(!runtime.data.join("service-private.json").exists());
    }
}
