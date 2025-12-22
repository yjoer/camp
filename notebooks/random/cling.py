# %%
import ctypes
import os
import shutil
import sys
from pathlib import Path

# %% [markdown]
# ## Cling

# %%
cling_path = shutil.which("cling")
cling_dir = (Path(cling_path) / ".." / "..").resolve()


# %%
libs = ["/bin/libclingJupyter", "/lib/libclingJupyter", "/libexec/lib/libclingJupyter"]

for lib in libs:
  for ext in [".dll", ".dylib", ".so"]:
    filename = cling_dir + lib + ext
    if not Path(filename).exists():
      continue

    cling = ctypes.CDLL(filename)


class my_void_p(ctypes.c_void_p):
  pass


cling.cling_create.restype = my_void_p
cling.cling_eval.restype = my_void_p

# %%
argv = [
  b"clingJupyter",
  b"-std=c++20",
  b"-I" + cling_dir.encode("utf-8") + b"/include/",
]
argv_type = ctypes.c_char_p * len(argv)
argc = len(argv)
llvm_dir = cling_dir.encode("utf-8")
r, w = os.pipe()

interpreter = cling.cling_create(
  ctypes.c_int(argc),
  argv_type(*argv),
  ctypes.c_char_p(llvm_dir),
  w,
)

# %%
code = """
#include <iostream>
"""

res = cling.cling_eval(interpreter, ctypes.c_char_p(code.encode("utf-8")))
ctypes.cast(res, ctypes.c_char_p).value.decode("utf-8")

# %%
code = """
gClingOpts->AllowRedefinition = 1;
int a = 1;
int b = 2;
a + b
"""

res = cling.cling_eval(interpreter, ctypes.c_char_p(code.encode("utf-8")))
ctypes.cast(res, ctypes.c_char_p).value.decode("utf-8")

# %%
code = """
int a = 1;
int b = 2;
std::cout << (a + b) << std::endl;
"""

res = cling.cling_eval(interpreter, ctypes.c_char_p(code.encode("utf-8")))
ctypes.cast(res, ctypes.c_char_p).value.decode("utf-8")

# %% [markdown]
# ## Pipes

# %%
ra, wa = os.pipe()
os.set_blocking(ra, False)

# %%
fd = os.fdopen(os.dup(wa), "w")
fd.write("hello")
fd.close()

# %%
if sys.platform == "win32":
  import msvcrt

  peek_named_pipe = ctypes.windll.kernel32.PeekNamedPipe
  peek_named_pipe.argtypes = [
    ctypes.wintypes.HANDLE,
    ctypes.c_void_p,
    ctypes.wintypes.DWORD,
    ctypes.POINTER(ctypes.wintypes.DWORD),
    ctypes.POINTER(ctypes.wintypes.DWORD),
    ctypes.POINTER(ctypes.wintypes.DWORD),
  ]
  peek_named_pipe.restype = ctypes.c_bool

  bytes_available = ctypes.wintypes.DWORD()
  status = peek_named_pipe(
    ctypes.wintypes.HANDLE(msvcrt.get_osfhandle(ra)),
    None,
    0,
    None,
    ctypes.byref(bytes_available),
    None,
  )

  print(status)
  print(bytes_available.value)

# %%
os.read(ra, 5)

# %%
if sys.platform == "win32":
  get_osfhandle = ctypes.windll.msvcrt._get_osfhandle
  get_osfhandle.argtypes = [ctypes.c_int]
  get_osfhandle.restype = ctypes.c_int
  print(get_osfhandle(ra))

# %%
