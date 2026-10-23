import numpy as np
import torch

from knee_abnormality.data import dataset
from knee_abnormality.data.preprocessing import preparation

POOL_SLOTS = 64
POOL_GRANULE = 2**22
POOL_BYTES = 3 * 2**28


class BufferPool:
    def __init__(self, flags: np.ndarray, offset: int) -> None:
        self.flags = flags
        self.offset = offset
        self.storages: list[torch.UntypedStorage | None] = [None] * len(flags)
        self.slots: dict[int, int] = {}
        self.pending: set[int] = set()

    def _storage(self, size: int) -> torch.UntypedStorage | None:
        free = np.flatnonzero(self.flags == 0)
        if not len(free):
            return None
        sizes = [
            0 if self.storages[slot] is None else self.storages[slot].nbytes()
            for slot in free
        ]
        fitting = [(held, slot) for held, slot in zip(sizes, free) if held >= size]
        if fitting:
            slot = min(fitting)[1]
        else:
            slot = free[int(np.argmin(sizes))]
            previous = self.storages[slot]
            rounded = -(-size // POOL_GRANULE) * POOL_GRANULE
            held = sum(
                storage.nbytes() for storage in self.storages if storage is not None
            ) - (0 if previous is None else previous.nbytes())
            if held + rounded > POOL_BYTES:
                return None
            storage = torch.UntypedStorage._new_shared(rounded)
            if previous is not None:
                del self.slots[previous.data_ptr()]
            self.storages[slot] = storage
            self.slots[storage.data_ptr()] = self.offset + int(slot)
        self.flags[slot] = 1
        self.pending.add(int(slot))
        return self.storages[slot]

    def empty(self, count: int, dtype: torch.dtype) -> torch.Tensor:
        size = count * torch.empty((), dtype=dtype).element_size()
        storage = self._storage(size) if dtype == torch.int16 and size else None
        if storage is None:
            storage = torch.UntypedStorage._new_shared(size)
        return torch.empty(0, dtype=dtype).set_(storage, 0, (count,))

    def batch_slots(self, buffers: list[torch.Tensor]) -> list[int]:
        slots = [
            self.slots.get(buffer.untyped_storage().data_ptr(), -1)
            for buffer in buffers
        ]
        for slot in self.pending - {slot - self.offset for slot in slots}:
            self.flags[slot] = 0
        self.pending.clear()
        return slots


_POOL: BufferPool | None = None


def buffer_flags(workers: int) -> np.ndarray:
    return torch.zeros((workers, POOL_SLOTS), dtype=torch.uint8).share_memory_().numpy()


def install_buffer_pool(flags: np.ndarray, worker_id: int) -> None:
    global _POOL
    _POOL = BufferPool(flags[worker_id], worker_id * POOL_SLOTS)
    preparation.shared_empty = _POOL.empty


def collate_pooled(items: list[dict]) -> dict:
    batch = dataset.collate_studies(items)
    batch["buffer_slots"] = (
        [-1] * len(batch["frame_buffers"])
        if _POOL is None
        else _POOL.batch_slots(batch["frame_buffers"])
    )
    return batch


class PinnedBuffers:
    def __init__(self) -> None:
        self.held: dict[int, tuple[torch.UntypedStorage, int, bool]] = {}

    def hold(self, slot: int, buffer: torch.Tensor, pin: bool) -> None:
        storage = buffer.untyped_storage()
        current = self.held.get(slot)
        if current is not None and current[1] == storage.data_ptr():
            return
        if current is not None and current[2]:
            torch.cuda.cudart().cudaHostUnregister(current[1])
        pinned = (
            pin
            and torch.cuda.cudart().cudaHostRegister(
                storage.data_ptr(), storage.nbytes(), 0
            )
            == torch.cuda.cudart().cudaError.success
        )
        self.held[slot] = (storage, storage.data_ptr(), pinned)

    def close(self) -> None:
        for _, pointer, pinned in self.held.values():
            if pinned:
                torch.cuda.cudart().cudaHostUnregister(pointer)
        self.held.clear()
