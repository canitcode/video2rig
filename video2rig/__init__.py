"""video2rig: a Seedance clip of your own rigged model -> an animation of that rig (see README.md)."""
import os

# On an Apple-silicon Mac the reader runs on Metal (video2rig/reader/device.py); an op Metal lacks falls back to the
# processor. PyTorch reads this when it loads, so it is set here, before any module imports torch.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
