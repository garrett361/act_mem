import torch
import torch.nn as nn
import act_mem

meta = torch.device("meta")
DIM = 8


class TestSavedBwdCaptureContext:
    def test_meta(self) -> None:
        t = torch.randn(1, DIM, device=meta, requires_grad=True)
        lin = nn.Linear(DIM, DIM, device=meta)
        with act_mem.SavedBwdCaptureContext(ignored_tensors=lin.parameters()) as saved:
            out = lin(t)
        assert saved.saved_tensor_mem == t.numel() * t.element_size()

    def test_meta_amp(self) -> None:
        t = torch.randn(1, DIM, device=meta, requires_grad=True)
        lin = nn.Linear(DIM, DIM, device=meta)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with act_mem.SavedBwdCaptureContext() as saved:
                out = lin(t)
        assert saved.saved_tensor_mem == t.numel() * t.element_size()
