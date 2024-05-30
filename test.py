import pytest
import torch
import torch.nn as nn

import act_mem
import layers

DIM = 128
SEQ_LEN = 64
N_HEADS = 4
BATCH_SIZE = 2

ACT_FNS = (
    nn.ELU,
    nn.GELU,
    nn.Hardshrink,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.Hardtanh,
    nn.LeakyReLU,
    nn.ReLU,
    nn.SELU,
    nn.SiLU,
    nn.Sigmoid,
    nn.Tanh,
)
ZERO_MEM_ACT_FNS = (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.RReLU)


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

    @pytest.mark.parametrize("act_fn_cls", ACT_FNS)
    def test_mlp(self, act_fn_cls) -> None:
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
        mlp = nn.Sequential(
            nn.Linear(DIM, expansion_factor * DIM),
            act_fn_cls(),
            nn.Linear(expansion_factor * DIM, DIM),
        )
        with act_mem.SavedBwdCaptureContext(ignored_tensors=mlp.parameters()) as saved:
            _ = mlp(inputs)

        # Compare measured memory against expected
        first_lin_input_mem = inputs.numel() * inputs.element_size()
        second_lin_input_mem = expansion_factor * inputs.numel() * inputs.element_size()
        # Only GeLU should cost additional activation memory
        activation_input_mem = (
            0 if act_fn_cls in (nn.ReLU, nn.Tanh, nn.Sigmoid) else second_lin_input_mem
        )

        expected_mem = first_lin_input_mem + second_lin_input_mem + activation_input_mem
        assert saved.saved_tensor_mem == expected_mem

    @pytest.mark.parametrize("act_fn_cls", ACT_FNS)
    def test_mlp_amp(self, act_fn_cls) -> None:
        """
        Similar story with AMP:
        """
        inputs = torch.randn(1, DIM, requires_grad=True)
        expansion_factor = 4
        mlp = nn.Sequential(
            nn.Linear(DIM, expansion_factor * DIM),
            act_fn_cls(),
            nn.Linear(expansion_factor * DIM, DIM),
        )
        dtype = torch.bfloat16
        with torch.autocast(device_type="cpu", dtype=dtype):
            with act_mem.SavedBwdCaptureContext(ignored_tensors=mlp.parameters()) as saved:
                _ = mlp(inputs)

        # Compare measured memory against expected
        amp_weight_mem = 2 * expansion_factor * DIM**2 * dtype.itemsize
        first_lin_input_mem = inputs.numel() * dtype.itemsize
        second_lin_input_mem = expansion_factor * inputs.numel() * dtype.itemsize
        # Only GeLU should cost additional activation memory
        activation_input_mem = (
            0 if act_fn_cls in (nn.ReLU, nn.Tanh, nn.Sigmoid) else second_lin_input_mem
        )

        expected_mem = (
            amp_weight_mem + first_lin_input_mem + second_lin_input_mem + activation_input_mem
        )
        assert saved.saved_tensor_mem == expected_mem


class TestLayers:
    def test_mlp(self) -> None:
        mlp = layers.MLP(d_model=DIM)
        inputs = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
        outputs = mlp(inputs)
        assert outputs.shape == inputs.shape

    def test_attention(self) -> None:
        attn = layers.Attention(d_model=DIM, n_heads=N_HEADS)
        inputs = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
        outputs = attn(inputs)
        assert outputs.shape == inputs.shape

    def test_block(self) -> None:
        block = layers.Block(d_model=DIM, n_heads=N_HEADS)
        inputs = torch.randn(BATCH_SIZE, SEQ_LEN, DIM)
        outputs = block(inputs)
        assert outputs.shape == inputs.shape
