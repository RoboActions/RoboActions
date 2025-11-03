"""Adds NumPy array support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""

import functools

import msgpack
import numpy as np


def _is_torch_tensor(x):
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    else:
        import torch
        return isinstance(x, torch.Tensor)


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    # torch.Tensor support (lazy import so torch is optional)
    if _is_torch_tensor(obj):
        import torch

        t: "torch.Tensor" = obj
        # Preserve metadata
        device = str(t.device)
        requires_grad = bool(t.requires_grad)

        # Move to CPU and convert to NumPy for a portable byte representation
        np_arr = t.detach().cpu().numpy()
        if np_arr.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported torch dtype (via numpy): {np_arr.dtype}")

        return {
            b"__torch_tensor__": True,
            b"data": np_arr.tobytes(),
            b"dtype": np_arr.dtype.str,
            b"shape": np_arr.shape,
            b"device": device,
            b"requires_grad": requires_grad,
        }

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__torch_tensor__" in obj:
        # Reconstruct as torch.Tensor if torch is available; otherwise fall back to NumPy array
        try:
            import torch
        except Exception:
            return np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"]))\
                .reshape(obj[b"shape"])  # NumPy fallback

        # Create a writable NumPy array copy (torch.from_numpy requires writable memory)
        np_arr = np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"]))
        np_arr = np_arr.reshape(obj[b"shape"]).copy()
        t = torch.from_numpy(np_arr)

        # Restore requires_grad if set
        if obj.get(b"requires_grad", False):
            t.requires_grad_(True)

        # Best-effort device restoration
        device_str = obj.get(b"device", b"cpu")
        try:
            device_decoded = device_str.decode() if isinstance(device_str, (bytes, bytearray)) else str(device_str)
            if device_decoded != "cpu":
                t = t.to(device_decoded)
        except Exception:
            # If device move fails (e.g., no CUDA available), keep on CPU
            pass

        return t

    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
