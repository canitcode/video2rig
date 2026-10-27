"""Which device the reader trains and reads on, and the mixed-precision context that goes with it.

The tool was written for NVIDIA cards (CUDA, bfloat16). This picks, in order: CUDA when PyTorch has it; Apple's
Metal (MPS) on an Apple-silicon Mac; else the processor. `--cpu` or V2R_CPU=1 forces the processor.

    pick_device(cpu=False) -> torch.device
    autocast(dev)          -> the `with` context for a forward pass on that device:
                              CUDA: bfloat16 (RTX 30 series and newer do it natively, as the README says)
                              MPS:  bfloat16 when this PyTorch/macOS can autocast to it, else float32 (no autocast;
                                    float16 is not used by default: training here has no loss scaler, so float16
                                    gradients could underflow). V2R_MPS_DTYPE=bfloat16|float16|float32 overrides.
                              CPU:  float32 (no autocast)

An op the Metal backend lacks runs on the processor instead (PYTORCH_ENABLE_MPS_FALLBACK=1, set in video2rig/__init__.py
before PyTorch loads), slower but correct.
"""
import contextlib
import os

_MPS_DTYPE = None


def pick_device(cpu=False):
    import torch
    if cpu or os.environ.get('V2R_CPU'):
        return torch.device('cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def describe(dev):
    """One line for the log: 'cuda (NVIDIA GeForce RTX 4070)', 'mps (Apple Metal, bfloat16)', 'cpu (16 threads)'."""
    import torch
    if dev.type == 'cuda':
        return f'cuda ({torch.cuda.get_device_name(dev)})'
    if dev.type == 'mps':
        dt = _mps_dtype()
        return f'mps (Apple Metal, {str(dt).replace("torch.", "") if dt is not None else "float32"})'
    return f'cpu ({torch.get_num_threads()} threads)'


def _mps_dtype():
    """The autocast dtype for MPS, probed once: bfloat16 if a small matmul autocasts to it here, else None (float32)."""
    global _MPS_DTYPE
    if _MPS_DTYPE is None:
        import torch
        want = os.environ.get('V2R_MPS_DTYPE', 'bfloat16').lower()
        if want in ('float32', 'fp32', 'none', 'off'):
            _MPS_DTYPE = False
        else:
            dt = torch.float16 if want in ('float16', 'fp16', 'half') else torch.bfloat16
            try:
                with torch.autocast('mps', dtype=dt):
                    a = torch.ones(8, 8, device='mps')
                    ok = (a @ a).dtype == dt
                _MPS_DTYPE = dt if ok else False
            except Exception:
                _MPS_DTYPE = False
    return _MPS_DTYPE or None


def autocast(dev):
    import torch
    if dev.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    if dev.type == 'mps':
        dt = _mps_dtype()
        if dt is not None:
            return torch.autocast('mps', dtype=dt)
    return contextlib.nullcontext()


def interpolate_area(x, **kw):
    """F.interpolate(mode='area'): on MPS the adaptive pool behind it refuses sizes that do not divide (PyTorch issue
    96056), and the CPU fallback does not cover it because the kernel exists and raises; so on MPS it runs on the
    processor and comes back. Elsewhere it is the plain call."""
    import torch.nn.functional as F
    if x.device.type == 'mps':
        return F.interpolate(x.cpu(), mode='area', **kw).to(x.device)
    return F.interpolate(x, mode='area', **kw)
