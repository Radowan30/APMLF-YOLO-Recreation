"""conftest.py — pytest sys.path configuration for APMLF-YOLO tests.

Problem: running pytest from the project root adds '' (CWD) to sys.path.
The CWD contains 'ultralytics/' which Python treats as a namespace package,
shadowing the editable-installed ultralytics package.

Fix:
  1. Add the ultralytics REPO ROOT to sys.path so 'import ultralytics' resolves
     to ultralytics/ultralytics/ (the real package with __init__.py).
  2. Add the ultralytics PACKAGE DIR to sys.path so tests can use the short
     form 'from nn.modules.X import Y' without the ultralytics. prefix.
  3. Remove '' (CWD) and the project root from sys.path to prevent the
     namespace package shadow.

After this conftest.py runs, both import styles work in all test files:
  from ultralytics.nn.modules.cam_light import NAMChannelAtt   # full path
  from nn.modules.cam_light import NAMChannelAtt                # short path
"""
import sys
import os

# Absolute paths
_here = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.dirname(_here)                       # APMLF_YOLO Implementation/
_repo_root = os.path.join(_proj_root, "ultralytics")      # ultralytics/ git repo
_pkg_dir = os.path.join(_repo_root, "ultralytics")        # ultralytics/ultralytics/

# Remove CWD and project root — they cause the namespace package shadow
for _path in ("", _proj_root):
    while _path in sys.path:
        sys.path.remove(_path)

# Insert repo root first (resolves 'import ultralytics' to the real package)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Insert package dir second (resolves 'from nn.modules.X import' short form)
if _pkg_dir not in sys.path:
    sys.path.insert(1, _pkg_dir)
