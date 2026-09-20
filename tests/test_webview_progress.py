"""Closing the bootstrap window exits promptly, leaving the parent Job to clean up."""
import ctypes as c
from ctypes import wintypes as w
from pathlib import Path
import subprocess
import sys
import time
import unittest


@unittest.skipUnless(sys.platform=='win32','Native Windows UI')
class ProgressTests(unittest.TestCase):
    def test_close_cancels_only_owned_window(self):
        user=c.WinDLL('user32')
        callback=c.WINFUNCTYPE(w.BOOL,w.HWND,w.LPARAM)
        user.EnumWindows.argtypes=[callback,w.LPARAM]
        user.GetWindowThreadProcessId.argtypes=[w.HWND,c.POINTER(w.DWORD)]
        user.PostMessageW.argtypes=[w.HWND,w.UINT,w.WPARAM,w.LPARAM]
        root=Path(__file__).resolve().parents[1]
        script='import time; from webview_progress import run_with_progress; run_with_progress(lambda update: time.sleep(60))'
        # Bypass venv's Windows redirector so the owned PID is the GUI process.
        child=subprocess.Popen([sys._base_executable,'-c',script],cwd=root/'scripts',creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline=time.monotonic()+10;owned=[]
            @callback
            def inspect(hwnd,_):
                pid=w.DWORD();user.GetWindowThreadProcessId(hwnd,c.byref(pid))
                if pid.value==child.pid:owned.append(hwnd)
                return True
            while not owned and child.poll() is None and time.monotonic()<deadline:
                user.EnumWindows(inspect,0);time.sleep(.05)
            self.assertTrue(owned,'Progress window was not created')
            user.PostMessageW(owned[0],0x10,0,0)
            self.assertEqual(child.wait(timeout=5),2)
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=5)


if __name__=='__main__':unittest.main()
