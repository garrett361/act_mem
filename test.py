import pytest
import torch
import torch.nn as nn

import act_mem
import layers

DIM = 128
SEQ_LEN = 64
N_HEADS = 4
BATCH_SIZE = 2

ZERO_MEM_ACT_FNS = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU(inplace=True)]
ALL_ACT_FNS = ZERO_MEM_ACT_FNS + [
    nn.ELU(),
    nn.GELU(),
    nn.Hardshrink(),
    nn.Hardsigmoid(),
    nn.Hardswish(),
    nn.Hardtanh(),
    nn.LeakyReLU(),
    nn.SELU(),
    nn.SiLU(),
]


class TestSavedBwdCaptureContext:
    def test_linear(self) -> None:
        """
        Test simple linear layer. The inputs should be saved for backwards
        """
        inputs = torch.randn(1, DIM, requires_grad=True)
        lin = nn.Linear(DIM, DIM)
        with act_mem.SavedBwdCaptureContext(ignored_tensors=lin.parameters()) as saved:
            _ = lin(inputs)
        assert saved.saved_tensor_mem == inputs.numel() * inputs.element_size()

    def test_linear_amp(self) -> None:
        """
        Test a linear layer with AMP. The saved tensors should now be a low-precision version of the
        inputs and the low-precision version of the weights version of the weights
        """
        inputs = torch.randn(1, DIM, requires_grad=True)
        lin = nn.Linear(DIM, DIM)
        dtype = torch.bfloat16
        with torch.autocast(device_type="cpu", dtype=dtype):
            with act_mem.SavedBwdCaptureContext(ignored_tensors=lin.parameters()) as saved:
                out = lin(inputs)
        assert (
            saved.saved_tensor_mem
            == out.numel() * out.element_size() + lin.weight.numel() * dtype.itemsize
        )

    @pytest.mark.parametrize("act_fn", ALL_ACT_FNS)
    def test_mlp(self, act_fn: nn.Module) -> None:
        """
        For the transformer MLP layer with a ReLU non-linearity, the initial inputs and the inputs
        to the final linear layer (which are four times as large) must always be saved. If the
        derivative of the activation function cannot be expressed in terms of the activation
        function's *outputs*, then the activation inputs must also be saved (which are again four
        times as large as the MLP's inputs). The MLP activation memory can be nearly halved by a
        choice of activation function.
        """
        inputs = torch.randn(1, DIM, requires_grad=True)
        expansion_factor = 4
        mlp = layers.MLP(d_model=DIM, act_fn=act_fn)
        with act_mem.SavedBwdCaptureContext(ignored_tensors=mlp.parameters()) as saved:
            _ = mlp(inputs)

        # Compare measured memory against expected
        first_lin_input_mem = inputs.numel() * inputs.element_size()
        second_lin_input_mem = expansion_factor * inputs.numel() * inputs.element_size()
        # Only some activations require additional activation memory
        activation_input_mem = 0 if act_fn in ZERO_MEM_ACT_FNS else second_lin_input_mem

        expected_mem = first_lin_input_mem + second_lin_input_mem + activation_input_mem
        assert saved.saved_tensor_mem == expected_mem, f"Failed on {act_fn=}"

    @pytest.mark.parametrize("act_fn", ALL_ACT_FNS)
    def test_mlp_amp(self, act_fn: nn.Module) -> None:
        """
        Similar story with AMP. The only changes come from the modified dtypes and needing to also
        save references to the low-precision weights in the Linear layers.
        """
        inputs = torch.randn(1, DIM, requires_grad=True)
        expansion_factor = 4
        mlp = layers.MLP(d_model=DIM, act_fn=act_fn)
        dtype = torch.bfloat16
        with torch.autocast(device_type="cpu", dtype=dtype):
            with act_mem.SavedBwdCaptureContext(ignored_tensors=mlp.parameters()) as saved:
                _ = mlp(inputs)

        # Compare measured memory against expected
        amp_weight_mem = 2 * expansion_factor * DIM**2 * dtype.itemsize
        first_lin_input_mem = inputs.numel() * dtype.itemsize
        second_lin_input_mem = expansion_factor * inputs.numel() * dtype.itemsize
        # Only some activations require additional activation memory
        activation_input_mem = 0 if act_fn in ZERO_MEM_ACT_FNS else second_lin_input_mem

        expected_mem = (
            amp_weight_mem + first_lin_input_mem + second_lin_input_mem + activation_input_mem
        )
        assert saved.saved_tensor_mem == expected_mem, f"Failed on {act_fn=}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
class TestCUDAMemReadings:
    def test(self):
        pass


class TestLayers:
    def test_mlp(self) -> None:
        mlp = layers.MLP(d_model=DIM, act_fn=nn.ReLU())
        inputs = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
        outputs = mlp(inputs)
        assert outputs.shape == inputs.shape

    def test_attention(self) -> None:
        attn = layers.Attention(d_model=DIM, n_heads=N_HEADS)
        inputs = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
        outputs = attn(inputs)
        assert outputs.shape == inputs.shape

    def test_block(self) -> None:
        block = layers.Block(d_model=DIM, n_heads=N_HEADS, act_fn=nn.ReLU())
        inputs = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
        outputs = block(inputs)
        assert outputs.shape == inputs.shape
