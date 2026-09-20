"""Small Win32 progress window, using only bundled Python's standard library."""
import ctypes as c
from ctypes import wintypes as w
import threading


def run_with_progress(operation):
    user=c.WinDLL('user32',use_last_error=True)
    kernel=c.WinDLL('kernel32',use_last_error=True)
    callback=c.WINFUNCTYPE(c.c_ssize_t,w.HWND,w.UINT,w.WPARAM,w.LPARAM)
    class WindowClass(c.Structure):
        _fields_=[('style',w.UINT),('proc',callback),('class_extra',c.c_int),('window_extra',c.c_int),
                  ('instance',w.HINSTANCE),('icon',w.HICON),('cursor',w.HANDLE),('brush',w.HBRUSH),
                  ('menu',w.LPCWSTR),('name',w.LPCWSTR)]
    user.DefWindowProcW.argtypes=[w.HWND,w.UINT,w.WPARAM,w.LPARAM];user.DefWindowProcW.restype=c.c_ssize_t
    user.CreateWindowExW.argtypes=[w.DWORD,w.LPCWSTR,w.LPCWSTR,w.DWORD,c.c_int,c.c_int,c.c_int,c.c_int,w.HWND,w.HMENU,w.HINSTANCE,c.c_void_p]
    user.CreateWindowExW.restype=w.HWND
    user.DestroyWindow.argtypes=[w.HWND]
    user.SetWindowTextW.argtypes=[w.HWND,w.LPCWSTR]
    user.SetTimer.argtypes=[w.HWND,c.c_size_t,w.UINT,c.c_void_p]
    user.SendMessageW.argtypes=[w.HWND,w.UINT,w.WPARAM,w.LPARAM];user.SendMessageW.restype=c.c_ssize_t
    user.LoadCursorW.argtypes=[w.HINSTANCE,w.LPCWSTR];user.LoadCursorW.restype=w.HANDLE
    user.RegisterClassW.argtypes=[c.POINTER(WindowClass)]
    user.GetMessageW.argtypes=[c.POINTER(w.MSG),w.HWND,w.UINT,w.UINT]
    user.DispatchMessageW.argtypes=[c.POINTER(w.MSG)];user.DispatchMessageW.restype=c.c_ssize_t
    user.MessageBoxW.argtypes=[w.HWND,w.LPCWSTR,w.LPCWSTR,w.UINT]
    kernel.GetModuleHandleW.argtypes=[w.LPCWSTR];kernel.GetModuleHandleW.restype=w.HMODULE
    state={'text':'正在连接微软下载服务…','done':False,'cancelled':False}
    label=None
    @callback
    def procedure(hwnd,message,wp,lp):
        if message==0x10:  # Close: parent Job Object stops extraction descendants.
            state['cancelled']=True;user.DestroyWindow(hwnd);return 0
        if message==2:
            user.PostQuitMessage(0);return 0
        if message==0x113:
            user.SetWindowTextW(label,state['text']+'\n首次使用需准备界面组件。关闭窗口可取消，下次启动继续下载。')
            if state['done']: user.DestroyWindow(hwnd)
            return 0
        return user.DefWindowProcW(hwnd,message,wp,lp)
    instance=kernel.GetModuleHandleW(None)
    wc=WindowClass(0,procedure,0,0,instance,None,None,6,None,'HyMTWebViewBootstrap')
    if not user.RegisterClassW(c.byref(wc)): raise c.WinError(c.get_last_error())
    window=user.CreateWindowExW(0,wc.name,'HyMT · 准备界面',0x10C80000,
        max(0,(user.GetSystemMetrics(0)-540)//2),max(0,(user.GetSystemMetrics(1)-150)//2),540,150,None,None,instance,None)
    if not window: raise c.WinError(c.get_last_error())
    label=user.CreateWindowExW(0,'STATIC',state['text'],0x50000000,20,24,495,80,window,None,instance,None)
    gdi=c.WinDLL('gdi32');gdi.GetStockObject.argtypes=[c.c_int];gdi.GetStockObject.restype=w.HANDLE
    user.SendMessageW(label,0x30,gdi.GetStockObject(17),1)
    def work():
        try: state['result']=operation(lambda text: state.update(text=text))
        except Exception as error: state['error']=error
        finally: state['done']=True
    threading.Thread(target=work,daemon=True).start()
    user.SetTimer(window,1,100,None)
    message=w.MSG()
    while user.GetMessageW(c.byref(message),None,0,0)>0:
        user.TranslateMessage(c.byref(message));user.DispatchMessageW(c.byref(message))
    if state['cancelled']: raise SystemExit(2)
    if 'error' in state: raise state['error']
    return state['result']
