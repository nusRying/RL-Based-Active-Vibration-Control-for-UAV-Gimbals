from __future__ import annotations

import importlib
import sys
import types


def strip_user_site_from_sys_path() -> None:
    """
    Remove user-site paths that can shadow conda/venv packages.
    """
    blocked_tokens = (
        "AppData\\Roaming\\Python",
        "/.local/lib/python",
    )
    sys.path = [
        p for p in sys.path if not any(token.lower() in p.lower() for token in blocked_tokens)
    ]


def patch_tensorboard_notf() -> None:
    """
    Some TensorBoard builds expect `tensorboard.compat.notf`.
    If missing, TensorBoard may force-import TensorFlow and fail in mixed stacks.
    Create a lightweight stub module to keep TensorBoard on tensorflow_stub path.
    """
    try:
        compat = importlib.import_module("tensorboard.compat")
    except Exception:
        return

    module_name = "tensorboard.compat.notf"
    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)
    sys.modules[module_name] = stub
    setattr(compat, "notf", stub)


def prepare_runtime_compat() -> None:
    strip_user_site_from_sys_path()
    patch_tensorboard_notf()
