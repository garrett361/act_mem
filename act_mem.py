import gc
from typing import Any, Iterable, Optional, Set, Union
from types import ModuleType

import torch


def get_tensor_bytes(tensor: torch.Tensor) -> int:
    """
    Returns the bytes of storage a given tensor takes up. If `tensor` is a view of a larger tensor,
    this function only returns the bytes associated with the view.
    """
    tensor_bytes = tensor.numel() * tensor.element_size()
    return tensor_bytes


def B_to_GiB(bytes: Union[int, float]) -> float:
    return bytes / 2**30


def get_tensor_GiB(tensor: torch.Tensor) -> float:
    """
    Returns the GiB of storage a given tensor takes up. If `tensor` is a view of a larger tensor,
    this function only returns the bytes associated with the view.
    """
    tensor_bytes = get_tensor_bytes(tensor)
    return B_to_GiB(tensor_bytes)


class AllocatedMemContext:
    """
    Context manager which computes the allocated GPU memory at context exit and the change between
    enter and exit.

    Only includes `allocated_bytes.all.`-prefixed keys in `memory_stats` with all readings converted
    to GiB.

    Example:

    ```python

    ```
    """

    def __init__(self, accel: ModuleType = torch.cuda) -> None:
        self.accel = accel
        self._enter_dict: dict[str, int] = {}
        self.mem_dict: dict[str, int] = {}

    def _get_allocation_dict(self) -> dict[str, int]:
        """Gets the memory allocation stats and does some cleanup."""
        self.accel.synchronize()
        gc.collect()
        key_prefix = "allocated_bytes.all."
        return {
            k.replace(key_prefix, ""): v
            for k, v in self.accel.memory_stats().items()
            if key_prefix in k
        }

    def __enter__(self) -> "AllocatedMemContext":
        self._enter_dict = self._get_allocation_dict()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        exit_dict = self._get_allocation_dict()
        delta_dict = {
            k + "_delta": v - self._enter_dict[k] for k, v in exit_dict.items()
        }
        self.mem_dict = {**delta_dict, **exit_dict}


class SavedBwdCaptureContext:
    """
    Context manager which captures all tensors which are registered as being saved for backwards
    within the context window. Mostly intended for tracking activations.

    All saved tensors are stored in the `saved_tensor_dict` attr, which is an instance of torch's
    WeakTensorKeyDictionary with tensor/data_ptr key/value pairs. The GiB in memory of all saved
    tensors with unique storage can be accessed through `saved_tensor_GiB`.

    Use:
    ```
    model = ...
    with SavedBwdCaptureContext(ignored_tensors=model.parameters()) as bwd_capture:
        # Do some computation with `model` and capture saved tensors which are not model weights

    ```
    bwd_capture.saved_tensor_dict # WeakTensorKeyDictionary of all saved tensors.
    bwd_capture.saved_tensor_GiB # GiB from all saved tensors (activation memory).
    """

    def __init__(
        self,
        ignored_tensors: Optional[Iterable[torch.Tensor]] = None,
    ) -> None:
        self._ignored_data_ptr_set: Set[int] = {
            p.untyped_storage().data_ptr() for p in (ignored_tensors or tuple())
        }

        self.saved_tensor_dict = torch.utils.weak.WeakTensorKeyDictionary()

        def pack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            data_ptr = saved_tensor.untyped_storage().data_ptr()
            if saved_tensor.is_meta or data_ptr not in self._ignored_data_ptr_set:
                self.saved_tensor_dict[saved_tensor] = data_ptr
            return saved_tensor

        def unpack_hook(saved_tensor: torch.Tensor) -> torch.Tensor:
            return saved_tensor

        self._saved_tensors_hook = torch.autograd.graph.saved_tensors_hooks(
            pack_hook, unpack_hook
        )

    def __enter__(self) -> "SavedBwdCaptureContext":
        self._saved_tensors_hook.__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._saved_tensors_hook.__exit__(*args, **kwargs)

    @property
    def saved_tensor_mem(self) -> int:
        # Avoid double counting of any memory. Can do this by comparing pointers when the tensor
        # device isn't meta. When the device is meta, the data_ptr is always zero, though, so
        seen_storage: Set[int] = set()
        total_bytes = 0
        for tensor, data_ptr in self.saved_tensor_dict.items():
            if tensor.is_meta or data_ptr not in seen_storage:
                seen_storage.add(data_ptr)
                total_bytes += tensor.untyped_storage().nbytes()
        return total_bytes
