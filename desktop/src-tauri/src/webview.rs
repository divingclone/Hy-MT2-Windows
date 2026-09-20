//! Prefer shared Evergreen; bootstrap a verified app-local runtime only if absent.
use crate::{job, CREATE_NO_WINDOW};
use std::{
    fs,
    io::Write,
    os::windows::process::CommandExt,
    path::PathBuf,
    process::{Command, Stdio},
};

pub fn prepare() -> Result<bool, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let directory = exe.parent().ok_or("Missing application directory")?;
    let portable = directory.join("portable.json").is_file();
    let data = if portable {
        directory.join("data")
    } else {
        PathBuf::from(std::env::var_os("LOCALAPPDATA").ok_or("LOCALAPPDATA is missing")?)
            .join("io.github.divingclone.hymt")
    };
    std::env::set_var("WEBVIEW2_USER_DATA_FOLDER", data.join("webview"));
    // Probe the registered system runtime, independently of any stale inherited path.
    std::env::remove_var("WEBVIEW2_BROWSER_EXECUTABLE_FOLDER");
    let force_fallback = std::env::var("HYMT_WEBVIEW2_FORCE_FALLBACK").as_deref() == Ok("1");
    if !force_fallback && tauri::webview_version().is_ok() {
        return Ok(true);
    }
    let root = if cfg!(debug_assertions) {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    } else {
        directory.join("payload")
    };
    let bundled = root.join("runtime/webview2");
    let browser = if bundled.join("msedgewebview2.exe").is_file() {
        bundled
    } else {
        fs::create_dir_all(data.join("logs")).map_err(|e| e.to_string())?;
        let log =
            fs::File::create(data.join("logs/webview-bootstrap.log")).map_err(|e| e.to_string())?;
        let mut python = root.join("runtime/python/python.exe");
        if !python.is_file() {
            python = root.join("runtime/python/cpython-3.12-windows-x86_64-none/python.exe");
        }
        let mut child = Command::new(python)
            .args(["-E", "-s", "-X", "utf8"])
            .arg(root.join("scripts/prepare_webview_runtime.py"))
            .args(["--progress", "--wait-for-parent", "--cache"])
            .arg(data.join("webview-runtime"))
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(log)
            .spawn()
            .map_err(|e| format!("启动界面组件下载失败：{e}"))?;
        let _job = job::Job::attach(&mut child).map_err(|e| e.to_string())?;
        writeln!(child.stdin.take().ok_or("Missing bootstrap input")?, "go")
            .map_err(|e| e.to_string())?;
        let result = child.wait_with_output().map_err(|e| e.to_string())?;
        if result.status.code() == Some(2) {
            return Ok(false);
        }
        if !result.status.success() {
            return Err(format!(
                "无法准备界面组件。请检查网络和磁盘空间后重新启动，未完成的下载会继续。\n详情：{}",
                data.join("logs/webview-bootstrap.log").display()
            ));
        }
        let path = PathBuf::from(
            String::from_utf8(result.stdout)
                .map_err(|e| e.to_string())?
                .trim(),
        );
        if !path.is_absolute() || !path.join("msedgewebview2.exe").is_file() {
            return Err("界面组件目录无效".into());
        }
        path
    };
    if windows_version::OsVersion::current().build < 22000 {
        for sid in ["*S-1-15-2-2:(OI)(CI)(RX)", "*S-1-15-2-1:(OI)(CI)(RX)"] {
            let status = Command::new("icacls.exe")
                .arg(&browser)
                .args(["/grant", sid])
                .creation_flags(CREATE_NO_WINDOW)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map_err(|e| e.to_string())?;
            if !status.success() {
                return Err("无法设置界面组件的读取权限，请将程序解压到可写的本地目录。".into());
            }
        }
    }
    std::env::set_var("WEBVIEW2_BROWSER_EXECUTABLE_FOLDER", browser);
    Ok(true)
}

pub fn show_error(message: &str) {
    #[link(name = "user32")]
    extern "system" {
        fn MessageBoxW(window: isize, text: *const u16, caption: *const u16, flags: u32) -> i32;
    }
    let text: Vec<u16> = message.encode_utf16().chain(Some(0)).collect();
    let title: Vec<u16> = "HyMT · 启动失败".encode_utf16().chain(Some(0)).collect();
    unsafe {
        MessageBoxW(0, text.as_ptr(), title.as_ptr(), 0x10);
    }
}
