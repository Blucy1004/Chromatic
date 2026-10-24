from pathlib import Path
import ctypes
import importlib.metadata as metadata
import json
import os
import platform
import sys
import winreg

root=Path(__file__).resolve().parents[1]
class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),('total_phys',ctypes.c_ulonglong),
              ('avail_phys',ctypes.c_ulonglong),('total_page',ctypes.c_ulonglong),('avail_page',ctypes.c_ulonglong),
              ('total_virtual',ctypes.c_ulonglong),('avail_virtual',ctypes.c_ulonglong),('avail_extended',ctypes.c_ulonglong)]
memory=Memory();memory.length=ctypes.sizeof(memory)
assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
    cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
data={'cpu_name':cpu,'logical_processors':os.cpu_count(),'os':platform.platform(),'python':sys.version,
      'physical_memory_bytes':memory.total_phys,'available_memory_at_collection_bytes':memory.avail_phys,
      'python_packages':{name:metadata.version(name) for name in ['python-chess','numpy','matplotlib']},
      'compiler':'MSVC 19.51.36248.0; CMake Release /O2 /Ob2 /DNDEBUG; /fp:precise; C++20',
      'search_threads':1,'affinity_mask':2,'search_logical_processor_zero_based':1,
      'cim_inventory_unavailable':'read-only registry and Windows memory API used instead'}
(root/'outputs/chromatic-depth-analysis/environment.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(data,ensure_ascii=False))
