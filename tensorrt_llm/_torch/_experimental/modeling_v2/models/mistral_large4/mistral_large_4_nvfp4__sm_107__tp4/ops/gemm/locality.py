# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paradigm-B single-kernel die-aware Rubin uGPU locality map and dual-domain VMM placement.

On dual-die Rubin (VR200 / SM107: 2 locality domains of 100 SMs + 12 remainder SMs), a single
persistent kernel launched across the device reads ``%smid`` once at entry and maps itself to a
die-local logical CTA index (:meth:`LocalityMap.logical_map` or :attr:`LocalityMap.sm_table`).
Each die's SMs stream only weights physically mapped in their own locality domain's HBM4, with zero
fork/join stream overhead and bitwise-identical arithmetic.

Supports both:
* ``n_split``: splitting output rows (or FC1 interleaved ``2I`` rows / FC2 ``H`` rows) across the
  two locality domains, covering ``wq``, ``wo``, ``ix_wq_b``, ``dense0``, ``lm_head``, and MoE.
* ``expert_half``: splitting routed experts ``[0, E/2)`` onto domain 0 and ``[E/2, E)`` onto
  domain 1, preserving the exact per-expert TRTLLM-Gen layout even when ``I/2`` is not 128-aligned
  (e.g. TP16 ``I = 384 -> I/2 = 192``).

The SM -> domain map is attested by the CUDA 13.4+ driver via ``cuDevSmResourceSplit`` with
``CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID`` and an NVRTC ``%smid`` probe on per-domain green
contexts. VMM allocations use ``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`` and are verified
fail-closed with ``CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL``.
"""

from __future__ import annotations

import ctypes
import math
import threading
from dataclasses import dataclass
from typing import Literal, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - allows CPU contract helpers without torch
    torch = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_VMM_GRANULARITY",
    "DomainSlabAllocator",
    "LocalityMap",
    "PlacedSplit",
    "build_logical_map_host",
    "build_sm_table_host",
    "compute_granule_split_row",
    "plan_expert_half_bytes",
    "plan_n_split_bytes",
    "recommend_moe_locality_mode",
]

DEFAULT_VMM_GRANULARITY = 2 * 1024 * 1024  # 2 MiB on Rubin VR200
_SLAB_ALIGN = 256  # TMA base-pointer alignment

_PROBE_SRC = r"""
extern "C" __global__ void smid_probe(int* out) {
    if (threadIdx.x == 0) {
        unsigned s;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
        out[blockIdx.x] = (int)s;
    }
}
"""

_LIBCUDA = None
_CACHE_LOCK = threading.Lock()
_MAPS: dict[int, "LocalityMap"] = {}


def _drv():
    from cuda.bindings import driver

    return driver


def _ck(res, what: str):
    """cuda-python returns (CUresult, *values): raise on an error, return the values."""
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"locality: {what} failed: {err}")
    if isinstance(res, tuple):
        return res[1] if len(res) == 2 else res[1:]
    return None


def _libcuda():
    global _LIBCUDA
    if _LIBCUDA is None:
        _LIBCUDA = ctypes.CDLL("libcuda.so.1")
    return _LIBCUDA


def _alloc_prop(device: int, domain: int):
    d = _drv()
    prop = d.CUmemAllocationProp()
    prop.type = d.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = d.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    prop.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
    prop.location.localized.deviceId = device
    prop.location.localized.localityDomainId = domain
    return prop


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


# -----------------------------------------------------------------------------
# Pure-Python host planning helpers (CPU-testable without CUDA).
# -----------------------------------------------------------------------------


def compute_granule_split_row(n: int, row_bytes: int, granule: int) -> int | None:
    """Return the granule-aligned split row nearest ``n // 2`` inside ``(0, n)``, or None."""
    if n <= 1 or row_bytes <= 0 or granule <= 0:
        return None
    step = granule // math.gcd(granule, row_bytes)
    s = round((n / 2.0) / step) * step
    return s if 0 < s < n else None


def build_sm_table_host(domain_sms: Sequence[Sequence[int]], num_sms: int) -> list[int]:
    """Build host ``sm_table[num_sms]``: ``(domain << 16) | rank_in_domain``; ``-1`` = remainder."""
    table = [-1] * num_sms
    for dom_id, sms in enumerate(domain_sms):
        for rank_in_dom, sm in enumerate(sms):
            if not (0 <= sm < num_sms):
                raise ValueError(f"SM id {sm} out of range [0, {num_sms})")
            if table[sm] != -1:
                raise ValueError(f"SM id {sm} assigned to multiple locality domains")
            table[sm] = (dom_id << 16) | rank_in_dom
    return table


def build_logical_map_host(
    domain_sms: Sequence[Sequence[int]],
    remainder: Sequence[int],
    num_sms: int,
    per_domain: int,
    extra: int,
    *,
    allow_idle: bool = False,
) -> tuple[list[int], list[int]]:
    """Build host ``smid -> logical_cta_id`` permutation for a dual-die GPU.

    Logical blocks ``[d * per_domain, (d + 1) * per_domain)`` land exclusively on domain ``d``'s
    first ``per_domain`` SMs. Logical blocks ``[2 * per_domain, 2 * per_domain + extra)`` land on
    the leftover domain SMs (alternating domains) followed by the remainder SMs. When
    ``allow_idle=True`` and ``2 * per_domain + extra < num_sms``, remaining SMs map to ``-1``
    (idle early-exit CTAs).
    """
    if len(domain_sms) != 2:
        raise ValueError(f"locality: expected 2 domains, got {len(domain_sms)}")
    sms_per_domain = min(len(domain_sms[0]), len(domain_sms[1]))
    if per_domain < 0 or extra < 0:
        raise ValueError(f"locality: negative CTA count (per_domain={per_domain}, extra={extra})")
    if per_domain > sms_per_domain:
        raise ValueError(f"locality: {per_domain} CTAs per domain > {sms_per_domain} SMs")
    total_active = 2 * per_domain + extra
    if not allow_idle and total_active != num_sms:
        raise ValueError(f"locality: {2 * per_domain} + {extra} CTAs must cover the {num_sms} SMs")
    if total_active > num_sms:
        raise ValueError(f"locality: {2 * per_domain} + {extra} CTAs exceed the {num_sms} SMs")

    sm_domain = [-1] * num_sms
    for dom_id, sms in enumerate(domain_sms):
        for sm in sms:
            sm_domain[sm] = dom_id

    lmap = [-1] * num_sms
    for dom_id in range(2):
        for i, sm in enumerate(domain_sms[dom_id][:per_domain]):
            lmap[sm] = dom_id * per_domain + i

    left = [domain_sms[dom_id][per_domain:] for dom_id in range(2)]
    rest = [x for pair in zip(*left) for x in pair]
    if len(left[0]) > len(left[1]):
        rest.extend(left[0][len(left[1]) :])
    elif len(left[1]) > len(left[0]):
        rest.extend(left[1][len(left[0]) :])
    rest.extend(remainder)

    ext_dom: list[int] = []
    for i, sm in enumerate(rest[:extra]):
        lmap[sm] = 2 * per_domain + i
        ext_dom.append(sm_domain[sm])

    assigned = sorted(v for v in lmap if v >= 0)
    if assigned != list(range(total_active)):
        raise RuntimeError(
            f"locality: malformed logical map (got {len(assigned)} unique blocks, expected {total_active})"
        )
    return lmap, ext_dom


def plan_n_split_bytes(
    shape: tuple[int, ...],
    element_size: int,
    dim: int = 0,
    granule: int = DEFAULT_VMM_GRANULARITY,
) -> tuple[int, int, bool]:
    """Compute ``(half_bytes, padded_half_bytes, exact_granule)`` for splitting ``shape`` in two
    equal halves along ``dim``."""
    if dim < 0:
        dim += len(shape)
    if not (0 <= dim < len(shape)):
        raise ValueError(f"dim {dim} out of range for shape {shape}")
    if shape[dim] % 2 != 0:
        raise ValueError(f"axis {dim} extent {shape[dim]} is not divisible by 2")
    total_elems = math.prod(shape)
    half_bytes = (total_elems // 2) * element_size
    padded_half = _align_up(half_bytes, granule)
    return half_bytes, padded_half, (half_bytes % granule == 0)


def plan_expert_half_bytes(
    shape: tuple[int, ...],
    element_size: int,
    granule: int = DEFAULT_VMM_GRANULARITY,
) -> tuple[int, int, int, bool]:
    """Compute ``(split_expert, half_bytes, padded_half_bytes, exact_granule)`` for splitting an
    ``[E, ...]`` routed-expert tensor into ``E/2`` experts on domain 0 and ``E/2`` on domain 1."""
    if len(shape) < 2 or shape[0] % 2 != 0:
        raise ValueError(f"expert_half requires shape [E, ...] with even E, got {shape}")
    split_expert = shape[0] // 2
    half_bytes, padded_half, exact = plan_n_split_bytes(shape, element_size, dim=0, granule=granule)
    return split_expert, half_bytes, padded_half, exact


def recommend_moe_locality_mode(
    moe_inter: int,
    num_experts: int = 128,
) -> Literal["n_split", "none"]:
    """Choose the default Paradigm-B MoE locality mode for a per-rank intermediate width.

    * For ``moe_inter >= 512`` with 128-aligned half-width (TP4 ``I=1536``, TP8 ``I=768``),
      ``n_split`` places FC1 and FC2 halves in die-local HBM4 with deterministic 50/50 balance.
    * For narrow widths ``moe_inter < 512`` (TP16 ``I=384``), ``none`` stays lowest-latency while
      ``n_split`` and ``expert_half`` remain available when requested explicitly.
    """
    del num_experts
    if moe_inter >= 512 and (moe_inter // 2) % 128 == 0:
        return "n_split"
    return "none"


# -----------------------------------------------------------------------------
# CUDA Driver VMM mapping and per-domain slab allocator.
# -----------------------------------------------------------------------------


class _Mapping:
    """One virtual range backed by per-domain physical allocations, exposed to torch through
    ``__cuda_array_interface__``. Lives as long as a tensor that views it."""

    def __init__(self, device: int, pieces: Sequence[tuple[int, int]]) -> None:
        """``pieces``: ``(bytes, domain)`` in address order; each a multiple of allocation granularity."""
        d = _drv()
        self.device = device
        self.size = sum(n for n, _ in pieces)
        self.va = _ck(d.cuMemAddressReserve(self.size, 0, 0, 0), "cuMemAddressReserve")
        self.handles: list[tuple[object, int]] = []
        off = 0
        try:
            for nbytes, dom in pieces:
                prop = _alloc_prop(device, dom)
                h = _ck(d.cuMemCreate(nbytes, prop, 0), f"cuMemCreate(domain {dom})")
                self.handles.append((h, nbytes))
                _ck(d.cuMemMap(int(self.va) + off, nbytes, 0, h, 0), "cuMemMap")
                off += nbytes
            acc = d.CUmemAccessDesc()
            acc.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            acc.location.id = device
            acc.flags = d.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _ck(d.cuMemSetAccess(self.va, self.size, [acc], 1), "cuMemSetAccess")
        except Exception:
            self.close()
            raise
        self.__cuda_array_interface__ = {
            "shape": (self.size,),
            "typestr": "|u1",
            "data": (int(self.va), False),
            "version": 3,
            "strides": None,
            "stream": None,
        }

    def close(self) -> None:
        d = _drv()
        off = 0
        for h, n in self.handles:
            d.cuMemUnmap(int(self.va) + off, n)
            d.cuMemRelease(h)
            off += n
        self.handles = []
        if self.va is not None:
            d.cuMemAddressFree(self.va, self.size)
            self.va = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class PlacedSplit:
    """Two die-local halves of one tensor (domain 0 and domain 1).

    ``contiguous`` is non-None when ``half0`` and ``half1`` sit back-to-back in a single virtual
    range with no inter-domain padding (so stock consumers can read it directly as a plain tensor).
    When a half's byte size is not a multiple of the VMM granularity (e.g. TP16 ``fc2_s`` 9 MiB or
    TP8 ``wq`` 3 MiB), ``half0`` and ``half1`` are separate TMA-aligned views backed by domain-0
    and domain-1 physical pages, and ``half1_byte_offset`` gives the byte stride inside ``mapping``
    when they share a single VMM reservation.
    """

    half0: "torch.Tensor"
    half1: "torch.Tensor"
    contiguous: "torch.Tensor | None"
    mode: Literal["n_split", "expert_half"]
    split_index: int
    half1_byte_offset: int
    exact_granule: bool
    mapping: _Mapping | None = None


class DomainSlabAllocator:
    """Packs small per-domain tensors (such as FP8 R128c4 scales or sub-granule halves) into
    shared granule-aligned VMM slabs per locality domain, avoiding 2 MiB waste per small tensor.

    Every allocated slice is 256-byte aligned (for TMA) and fail-closed verified via
    :meth:`LocalityMap.domain_of`.
    """

    def __init__(self, locality_map: "LocalityMap", slab_bytes: int | None = None) -> None:
        self._lm = locality_map
        self._gran = locality_map.granularity
        default_slab = _align_up(16 * 1024 * 1024, self._gran)
        self._slab_bytes = default_slab if slab_bytes is None else _align_up(slab_bytes, self._gran)
        # Per domain: list of (mapping, flat uint8 tensor, cursor byte offset)
        self._slabs: list[list[tuple[_Mapping, "torch.Tensor", int]]] = [[], []]

    def _new_slab(self, domain: int, min_bytes: int) -> tuple[_Mapping, "torch.Tensor", int]:
        size = max(self._slab_bytes, _align_up(min_bytes, self._gran))
        m = _Mapping(self._lm.device, [(size, domain)])
        flat = torch.as_tensor(m, device=f"cuda:{self._lm.device}")
        self._lm._verify_range(flat.data_ptr(), size, domain, f"slab(domain {domain})")
        entry = (m, flat, 0)
        self._slabs[domain].append(entry)
        return entry

    def copy_to_domain(
        self, src: "torch.Tensor", domain: int, *, align: int = _SLAB_ALIGN
    ) -> "torch.Tensor":
        """Copy contiguous ``src`` into a TMA-aligned slice physically backed by ``domain``."""
        if not src.is_contiguous():
            src = src.contiguous()
        nbytes = src.numel() * src.element_size()
        if nbytes == 0:
            raise ValueError("DomainSlabAllocator: cannot place an empty tensor")
        if domain not in (0, 1):
            raise ValueError(f"DomainSlabAllocator: invalid domain {domain}")
        slabs = self._slabs[domain]
        idx = -1
        for i, (_, flat, cursor) in enumerate(slabs):
            start = _align_up(cursor, align)
            if start + nbytes <= flat.numel():
                idx = i
                break
        if idx < 0:
            self._new_slab(domain, nbytes)
            idx = len(slabs) - 1
        m, flat, cursor = slabs[idx]
        start = _align_up(cursor, align)
        view = flat[start : start + nbytes].view(src.dtype).view(src.shape)
        view.copy_(src)
        view._locality_mapping = m  # keep the slab alive as long as any view exists
        slabs[idx] = (m, flat, start + nbytes)
        self._lm._verify_range(view.data_ptr(), nbytes, domain, f"slab_slice(domain {domain})")
        return view


class LocalityMap:
    """The locality domains of one multi-die GPU. Build with :meth:`get` before CUDA-graph capture."""

    def __init__(self, device: int) -> None:
        if torch is None:
            raise RuntimeError("locality: PyTorch is required to probe a CUDA device")
        d = _drv()
        _ck(d.cuInit(0), "cuInit")
        dev = _ck(d.cuDeviceGet(device), "cuDeviceGet")
        attr = d.CUdevice_attribute
        self.device = device
        self.n_domains = int(
            _ck(
                d.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, dev),
                "domain count",
            )
        )
        self.sms_per_domain = int(
            _ck(
                d.cuDeviceGetAttribute(
                    attr.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_MULTIPROCESSOR_COUNT, dev
                ),
                "SMs per domain",
            )
        )
        self.num_sms = int(
            _ck(d.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "SMs")
        )
        if self.n_domains != 2:
            raise RuntimeError(
                f"locality: {self.n_domains} locality domains (this layer handles 2)"
            )
        self._domain_sms = self._probe(d, dev)
        dom = [-1] * self.num_sms
        for k, sms in enumerate(self._domain_sms):
            for s in sms:
                if dom[s] != -1:
                    raise RuntimeError(f"locality: SM {s} probed in two domains")
                dom[s] = k
        self.sm_domain: list[int] = dom
        self.remainder: list[int] = [s for s in range(self.num_sms) if dom[s] < 0]
        rt = _alloc_prop(device, 0)
        self.granularity = int(
            _ck(
                d.cuMemGetAllocationGranularity(
                    rt, d.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
                ),
                "granularity",
            )
        )
        self._sm_table: torch.Tensor | None = None
        self._streams: list[torch.cuda.ExternalStream] | None = None
        self._green: list[tuple[object, object]] = []
        self._slab: DomainSlabAllocator | None = None

    @classmethod
    def get(cls, device) -> "LocalityMap":
        """Return the cached :class:`LocalityMap` for ``device`` (built before graph capture)."""
        if torch is None:
            raise RuntimeError("locality: PyTorch is required")
        idx = torch.device(device).index if not isinstance(device, int) else device
        idx = torch.cuda.current_device() if idx is None else idx
        with _CACHE_LOCK:
            m = _MAPS.get(idx)
            if m is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "LocalityMap probes the device: build it before CUDA-graph capture"
                    )
                m = _MAPS[idx] = cls(idx)
        return m

    @classmethod
    def is_available(cls, device=None) -> tuple[bool, str]:
        """Return ``(True, '')`` when ``device`` exposes 2 locality domains and probes cleanly."""
        if torch is None or not torch.cuda.is_available():
            return False, "no CUDA"
        try:
            dev = torch.cuda.current_device() if device is None else device
            cls.get(dev)
            return True, ""
        except (ImportError, RuntimeError, OSError) as err:
            return False, str(err)

    def sms(self, domain: int) -> list[int]:
        return list(self._domain_sms[domain])

    @property
    def slab(self) -> DomainSlabAllocator:
        """Shared per-domain slab allocator for small tensors (scales, sub-granule weights)."""
        if self._slab is None:
            self._slab = DomainSlabAllocator(self)
        return self._slab

    @property
    def sm_table(self) -> "torch.Tensor":
        """int32 ``[num_sms]`` on device: ``(domain << 16) | rank_in_domain``; ``-1`` = remainder."""
        if self._sm_table is None:
            host = build_sm_table_host(self._domain_sms, self.num_sms)
            self._sm_table = torch.tensor(host, dtype=torch.int32, device=f"cuda:{self.device}")
        return self._sm_table

    def _probe(self, d, dev) -> list[list[int]]:
        """Split SMs by locality domain (strict, no backfill) and record ``%smid`` on each green
        context so the driver attests the hardware SM -> domain assignment."""
        from cuda.bindings import nvrtc

        full = _ck(
            d.cuDeviceGetDevResource(dev, d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
            "dev resource",
        )
        params = []
        for k in range(self.n_domains):
            p = d.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
            p.smCount = self.sms_per_domain
            p.flags = d.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
            p.localityDomainId = k
            params.append(p)
        res, _rem = _ck(
            d.cuDevSmResourceSplit(self.n_domains, full, 0, params), "cuDevSmResourceSplit"
        )
        major = _ck(
            d.cuDeviceGetAttribute(
                d.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
            ),
            "cc major",
        )
        minor = _ck(
            d.cuDeviceGetAttribute(
                d.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
            ),
            "cc minor",
        )
        prog = _ck(
            nvrtc.nvrtcCreateProgram(_PROBE_SRC.encode(), b"smid_probe.cu", 0, [], []),
            "nvrtcCreateProgram",
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
        if int(err) != 0:
            raise RuntimeError(f"locality: probe compile failed: {err}")
        n = _ck(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
        cubin = b" " * n
        _ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
        out: list[list[int]] = []
        nb = 4 * self.sms_per_domain
        torch.cuda.set_device(self.device)
        buf = torch.full((nb,), -1, dtype=torch.int32, device=f"cuda:{self.device}")
        for k in range(self.n_domains):
            desc = _ck(d.cuDevResourceGenerateDesc([res[k]], 1), "resource desc")
            g = _ck(
                d.cuGreenCtxCreate(desc, dev, d.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM),
                "green ctx",
            )
            st = _ck(
                d.cuGreenCtxStreamCreate(g, d.CUstream_flags.CU_STREAM_NON_BLOCKING, 0),
                "green stream",
            )
            ctx = _ck(d.cuCtxFromGreenCtx(g), "ctx from green ctx")
            _ck(d.cuCtxPushCurrent(ctx), "push ctx")
            try:
                mod = _ck(d.cuModuleLoadData(cubin), "probe module")
                fn = _ck(d.cuModuleGetFunction(mod, b"smid_probe"), "probe fn")
                buf.fill_(-1)
                torch.cuda.synchronize(self.device)
                arg = ctypes.c_void_p(buf.data_ptr())
                _ck(
                    d.cuLaunchKernel(
                        fn, nb, 1, 1, 32, 1, 1, 0, st, ((arg,), (ctypes.c_void_p,)), 0
                    ),
                    "probe launch",
                )
                _ck(d.cuStreamSynchronize(st), "probe sync")
                sms = sorted({int(x) for x in buf.tolist() if x >= 0})
                d.cuModuleUnload(mod)
            finally:
                d.cuCtxPopCurrent()
                d.cuStreamDestroy(st)
                d.cuGreenCtxDestroy(g)
            if len(sms) != self.sms_per_domain:
                raise RuntimeError(
                    f"locality: domain {k} probed {len(sms)} SMs, expected {self.sms_per_domain}"
                )
            out.append(sms)
        return out

    def logical_map(
        self, per_domain: int, extra: int, *, allow_idle: bool = False
    ) -> tuple["torch.Tensor", list[int]]:
        """SM -> logical CTA index permutation on device (``int32 [num_sms]``)."""
        lmap, ext_dom = build_logical_map_host(
            self._domain_sms,
            self.remainder,
            self.num_sms,
            per_domain,
            extra,
            allow_idle=allow_idle,
        )
        return torch.tensor(lmap, dtype=torch.int32, device=f"cuda:{self.device}"), ext_dom

    def domain_streams(self) -> list["torch.cuda.ExternalStream"]:
        """One non-blocking stream per domain on a green context holding that domain's SMs."""
        if self._streams is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("locality.domain_streams: build before CUDA-graph capture")
            d = _drv()
            dev = _ck(d.cuDeviceGet(self.device), "cuDeviceGet")
            full = _ck(
                d.cuDeviceGetDevResource(dev, d.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
                "dev resource",
            )
            params = []
            for k in range(self.n_domains):
                p = d.CU_DEV_SM_RESOURCE_GROUP_PARAMS()
                p.smCount = self.sms_per_domain
                p.flags = d.CUdevSmResourceGroup_flags.CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID
                p.localityDomainId = k
                params.append(p)
            res, _rem = _ck(
                d.cuDevSmResourceSplit(self.n_domains, full, 0, params), "cuDevSmResourceSplit"
            )
            self._green = []
            st: list[torch.cuda.ExternalStream] = []
            for k in range(self.n_domains):
                desc = _ck(d.cuDevResourceGenerateDesc([res[k]], 1), "resource desc")
                g = _ck(
                    d.cuGreenCtxCreate(
                        desc, dev, d.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
                    ),
                    "green ctx",
                )
                s_ = _ck(
                    d.cuGreenCtxStreamCreate(g, d.CUstream_flags.CU_STREAM_NON_BLOCKING, 0),
                    "green stream",
                )
                self._green.append((g, s_))
                st.append(torch.cuda.ExternalStream(int(s_), device=f"cuda:{self.device}"))
            self._streams = st
        return self._streams

    def domain_of(self, ptr: int) -> int:
        """Locality domain ordinal of a device address (-1 if not placed in one domain)."""
        d = _drv()
        lib = _libcuda()
        out = ctypes.c_int(-9)
        attr = int(d.CUpointer_attribute.CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL)
        err = lib.cuPointerGetAttribute(ctypes.byref(out), ctypes.c_int(attr), ctypes.c_uint64(ptr))
        if err != 0:
            raise RuntimeError(
                f"locality: cuPointerGetAttribute(LOCALITY_DOMAIN_ORDINAL) failed: {err}"
            )
        return int(out.value)

    def _verify_range(self, ptr: int, nbytes: int, expected_domain: int, what: str) -> None:
        """Fail-closed check that the start and end bytes of a placed range sit in ``expected_domain``."""
        for addr in (ptr, ptr + nbytes - 1):
            actual = self.domain_of(addr)
            if actual != expected_domain:
                raise RuntimeError(
                    f"locality.{what}: byte {addr:#x} landed in domain {actual}, "
                    f"expected {expected_domain}"
                )

    def place_split(self, t: "torch.Tensor", split_row: int) -> "torch.Tensor":
        """Exact contiguous split-mapped placement along dim 0: rows ``[0, split_row)`` in domain 0
        and ``[split_row, N)`` in domain 1 inside one virtual range (same shape and layout)."""
        if not t.is_contiguous():
            raise ValueError("locality.place_split: needs a contiguous tensor")
        row_bytes = t[0].numel() * t.element_size() if t.dim() > 1 else t.element_size()
        head = split_row * row_bytes
        total = t.numel() * t.element_size()
        if not 0 < head < total or head % self.granularity:
            raise ValueError(
                f"locality.place_split: rows [0, {split_row}) = {head} B must be a granule multiple "
                f"({self.granularity} B) inside the {total} B tensor"
            )
        tail = _align_up(total - head, self.granularity)
        m = _Mapping(self.device, [(head, 0), (tail, 1)])
        flat = torch.as_tensor(m, device=f"cuda:{self.device}")
        out = flat[:total].view(t.dtype).view(t.shape)
        out.copy_(t)
        out._locality_mapping = m
        self._verify_range(out.data_ptr(), head, 0, "place_split[0]")
        self._verify_range(out.data_ptr() + head, total - head, 1, "place_split[1]")
        return out

    def replace_param_(self, param: "torch.nn.Parameter", split_row: int) -> "torch.nn.Parameter":
        """Re-place a parameter in place with :meth:`place_split` (before CUDA-graph capture)."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("locality.replace_param_: not during CUDA-graph capture")
        param.data = self.place_split(param.data, split_row)
        return param

    def split_rows(self, t: "torch.Tensor", dim: int = 1) -> "torch.Tensor":
        """Exact contiguous ``[2, ...]`` placement: half ``k`` along ``dim`` in domain ``k``."""
        if t.element_size() != 1 or not t.is_contiguous():
            raise ValueError("locality.split_rows: needs a contiguous 1-byte tensor")
        n = t.shape[dim]
        if n % 2:
            raise ValueError(f"locality.split_rows: {n} rows don't split in halves")
        half_bytes = t.numel() // 2
        if half_bytes % self.granularity:
            raise ValueError(
                f"locality.split_rows: a half is {half_bytes} B, not a multiple of the "
                f"{self.granularity} B granularity"
            )
        m = _Mapping(self.device, [(half_bytes, 0), (half_bytes, 1)])
        flat = torch.as_tensor(m, device=f"cuda:{self.device}")
        shape = (2,) + tuple(t.shape[:dim]) + (n // 2,) + tuple(t.shape[dim + 1 :])
        out = flat.view(torch.uint8).view(shape)
        src = t.view(torch.uint8)
        for k in range(2):
            out[k].copy_(src.narrow(dim, k * (n // 2), n // 2))
        out._locality_mapping = m
        for k in range(2):
            self._verify_range(out[k].data_ptr(), half_bytes, k, f"split_rows[{k}]")
        return out

    def split_rows_padded(
        self,
        t: "torch.Tensor",
        dim: int = 0,
        *,
        mode: Literal["n_split", "expert_half"] = "n_split",
        slab: DomainSlabAllocator | None = None,
    ) -> PlacedSplit:
        """Split ``t`` along ``dim`` into equal halves placed physically in domain 0 and domain 1.

        * When ``half_bytes`` is a multiple of the VMM granularity (e.g. ``ix_wq_b`` 12 MiB, TP8
          ``fc1_w/s`` & ``fc2_w/s``, TP16 ``fc1_w/s`` & ``fc2_w``), both halves occupy a single
          back-to-back virtual range with zero padding and ``PlacedSplit.contiguous`` is populated.
        * When ``half_bytes < granularity`` and ``slab`` is provided (e.g. FP8 R128c4 scales),
          each half is packed into ``slab`` so zero 2 MiB granules are wasted.
        * Otherwise (e.g. TP16 ``fc2_s`` 9 MiB, TP8 ``wq``/``wo`` 3 MiB), a single VMM reservation
          maps ``[pad_up(half_bytes, gran) on domain 0, pad_up(half_bytes, gran) on domain 1]`` and
          returns TMA-aligned ``half0`` and ``half1`` views with fail-closed domain checks.
        """
        if not t.is_contiguous():
            t = t.contiguous()
        if dim < 0:
            dim += t.dim()
        n = t.shape[dim]
        if n % 2 != 0:
            raise ValueError(f"locality.split_rows_padded: dim {dim} extent {n} is odd")
        half_n = n // 2
        half_bytes, padded_half, exact = plan_n_split_bytes(
            tuple(t.shape), t.element_size(), dim=dim, granule=self.granularity
        )
        half_shape = tuple(t.shape[:dim]) + (half_n,) + tuple(t.shape[dim + 1 :])
        src0 = t.narrow(dim, 0, half_n).contiguous()
        src1 = t.narrow(dim, half_n, half_n).contiguous()

        if slab is not None and half_bytes < self.granularity:
            h0 = slab.copy_to_domain(src0, 0)
            h1 = slab.copy_to_domain(src1, 1)
            return PlacedSplit(
                half0=h0,
                half1=h1,
                contiguous=None,
                mode=mode,
                split_index=half_n,
                half1_byte_offset=0,
                exact_granule=False,
                mapping=None,
            )

        m = _Mapping(self.device, [(padded_half, 0), (padded_half, 1)])
        flat = torch.as_tensor(m, device=f"cuda:{self.device}")
        h0 = flat[:half_bytes].view(t.dtype).view(half_shape)
        h1 = flat[padded_half : padded_half + half_bytes].view(t.dtype).view(half_shape)
        h0.copy_(src0)
        h1.copy_(src1)
        h0._locality_mapping = m
        h1._locality_mapping = m
        self._verify_range(h0.data_ptr(), half_bytes, 0, "split_rows_padded[0]")
        self._verify_range(h1.data_ptr(), half_bytes, 1, "split_rows_padded[1]")

        contiguous = None
        if exact:
            full_shape = tuple(t.shape) if dim == 0 else ((2,) + half_shape)
            contiguous = flat[: 2 * half_bytes].view(t.dtype).view(full_shape)
            contiguous._locality_mapping = m
        return PlacedSplit(
            half0=h0,
            half1=h1,
            contiguous=contiguous,
            mode=mode,
            split_index=half_n,
            half1_byte_offset=padded_half,
            exact_granule=exact,
            mapping=m,
        )

    def place_expert_half(self, t: "torch.Tensor", *, allow_padded: bool = True) -> PlacedSplit:
        """Place ``[E, ...]`` routed-expert weights/scales with experts ``[0, E/2)`` in domain 0
        and ``[E/2, E)`` in domain 1.

        For ML4 TP8 (``I=768``), all four tensors (``fc1_w``, ``fc1_s``, ``fc2_w``, ``fc2_s``) hit
        exact 2 MiB boundaries so ``PlacedSplit.contiguous`` is an exact in-place ``[E, ...]`` view.
        For ML4 TP16 (``I=384``), ``fc1_w``, ``fc1_s``, and ``fc2_w`` hit exact 2 MiB boundaries,
        while ``fc2_s`` (9 MiB/half) uses padded dual-domain VMM (10 MiB/half) when
        ``allow_padded=True``.
        """
        split_expert, half_bytes, _, exact = plan_expert_half_bytes(
            tuple(t.shape), t.element_size(), granule=self.granularity
        )
        if not exact and not allow_padded:
            raise ValueError(
                f"locality.place_expert_half: experts [0, {split_expert}) = {half_bytes} B is not "
                f"a multiple of {self.granularity} B and allow_padded=False"
            )
        return self.split_rows_padded(t, dim=0, mode="expert_half")

    def place_n_split(
        self,
        weight: "torch.Tensor",
        weight_scale: "torch.Tensor",
        *,
        use_slab: bool = True,
    ) -> tuple[PlacedSplit, PlacedSplit]:
        """Place an FP8 linear's ``weight [N, K]`` and R128c4 ``weight_scale`` with rows
        ``[0, N/2)`` in domain 0 and ``[N/2, N)`` in domain 1. Requires ``(N // 2) % 128 == 0``."""
        if weight.dim() != 2 or not weight.is_contiguous():
            raise ValueError("locality.place_n_split: weight must be contiguous [N, K]")
        n, k = weight.shape
        if (n // 2) % 128 != 0 or k % 128 != 0:
            raise ValueError(
                f"locality.place_n_split: (N/2, K) = ({n // 2}, {k}) must be multiples of 128"
            )
        expected_sf = n * k // 32
        if weight_scale.numel() != expected_sf:
            raise ValueError(
                f"locality.place_n_split: weight_scale has {weight_scale.numel()} bytes, "
                f"expected {expected_sf}"
            )
        slab = self.slab if use_slab else None
        w_placed = self.split_rows_padded(weight, dim=0, mode="n_split", slab=slab)
        ws_2d = weight_scale.view(torch.uint8).reshape(n // 128, (k // 128) * 512)
        ws_placed = self.split_rows_padded(ws_2d, dim=0, mode="n_split", slab=slab)
        return w_placed, ws_placed
