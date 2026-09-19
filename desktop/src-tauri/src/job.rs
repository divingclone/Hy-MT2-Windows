//! Windows Job Object is intentionally native: the shell plugin alone cannot
//! guarantee descendant cleanup when the GUI is forcibly terminated.
use std::{io, os::windows::io::AsRawHandle, process::Child};
use windows_sys::Win32::{
    Foundation::{CloseHandle, HANDLE},
    System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectBasicAccountingInformation,
        JobObjectExtendedLimitInformation, QueryInformationJobObject, SetInformationJobObject,
        TerminateJobObject, JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    },
};

pub struct Job(HANDLE);
// An owned kernel handle can be transferred across threads. It is never cloned.
unsafe impl Send for Job {}

impl Job {
    pub fn attach(child: &mut Child) -> io::Result<Self> {
        unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() {
                let _ = child.kill();
                return Err(io::Error::last_os_error());
            }
            let job = Self(handle);
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                std::mem::size_of_val(&info) as u32,
            ) == 0
                || AssignProcessToJobObject(handle, child.as_raw_handle() as HANDLE) == 0
            {
                let error = io::Error::last_os_error();
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
            Ok(job)
        }
    }
    pub fn terminate(&self) {
        unsafe {
            TerminateJobObject(self.0, 1);
            // Wait for all descendants, including CUDA DLL users, before a
            // restart/update is allowed to replace their runtime files.
            for _ in 0..200 {
                let mut info: JOBOBJECT_BASIC_ACCOUNTING_INFORMATION = std::mem::zeroed();
                if QueryInformationJobObject(
                    self.0,
                    JobObjectBasicAccountingInformation,
                    &mut info as *mut _ as *mut _,
                    std::mem::size_of_val(&info) as u32,
                    std::ptr::null_mut(),
                ) == 0
                    || info.ActiveProcesses == 0
                {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
        }
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        os::windows::process::CommandExt,
        path::PathBuf,
        process::{Command, Stdio},
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE,
    };

    #[test]
    fn dropping_job_kills_worker_and_its_descendant() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let python = root.join("runtime/python/cpython-3.12-windows-x86_64-none/python.exe");
        if !python.exists() {
            eprintln!("Bundled runtime unavailable; lifecycle integration test skipped");
            return;
        }
        let mut child = Command::new(&python).args(["-u", "-c", "import sys,subprocess,time; sys.stdin.readline(); p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']); print(p.pid,flush=True); time.sleep(120)"])
            .creation_flags(0x08000000).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
        let job = Job::attach(&mut child).unwrap();
        writeln!(child.stdin.take().unwrap(), "go").unwrap();
        let mut pid = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut pid)
            .unwrap();
        unsafe {
            let descendant = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid.trim().parse().unwrap());
            assert!(!descendant.is_null());
            drop(job);
            // Windows may report exit code 0 for kill-on-close; liveness, not
            // the exit code, is the property under test.
            let _ = child.wait().unwrap();
            let result = WaitForSingleObject(descendant, 5000);
            CloseHandle(descendant);
            assert_eq!(result, 0, "descendant survived parent Job closure");
        }
    }
}
