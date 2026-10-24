"""Windows process-commit hard limit, plus cross-platform RSS checks elsewhere."""
import os
_job=None
def install(gigabytes):
    global _job
    if os.name!='nt':return {'hard_limit':False,'method':'RSS checks; Windows Job Object unavailable'}
    import ctypes as c
    from ctypes import wintypes as w
    class Basic(c.Structure):
        _fields_=[('process_time',c.c_longlong),('job_time',c.c_longlong),('flags',w.DWORD),
                  ('min_ws',c.c_size_t),('max_ws',c.c_size_t),('active',w.DWORD),
                  ('affinity',c.c_size_t),('priority',w.DWORD),('scheduling',w.DWORD)]
    class IO(c.Structure):_fields_=[(n,c.c_ulonglong) for n in ['read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes']]
    class Extended(c.Structure):_fields_=[('basic',Basic),('io',IO),('process_memory',c.c_size_t),('job_memory',c.c_size_t),('peak_process',c.c_size_t),('peak_job',c.c_size_t)]
    kernel=c.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.argtypes=[c.c_void_p,w.LPCWSTR];kernel.CreateJobObjectW.restype=w.HANDLE
    kernel.SetInformationJobObject.argtypes=[w.HANDLE,c.c_int,c.c_void_p,w.DWORD];kernel.SetInformationJobObject.restype=w.BOOL
    kernel.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];kernel.AssignProcessToJobObject.restype=w.BOOL
    kernel.GetCurrentProcess.restype=w.HANDLE
    handle=kernel.CreateJobObjectW(None,None)
    if not handle:raise c.WinError(c.get_last_error())
    limits=Extended();limits.basic.flags=0x100;limits.process_memory=int(gigabytes*1024**3)
    if not kernel.SetInformationJobObject(handle,9,c.byref(limits),c.sizeof(limits)):raise c.WinError(c.get_last_error())
    if not kernel.AssignProcessToJobObject(handle,kernel.GetCurrentProcess()):raise c.WinError(c.get_last_error())
    _job=handle
    return {'hard_limit':True,'method':'Windows Job Object process committed-memory limit','bytes':limits.process_memory}
