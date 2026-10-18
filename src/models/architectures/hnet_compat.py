"""
Capa de compatibilidad (shims) entre el código vendorizado de H-Net y las
versiones fijadas del entorno (PyTorch 2.5.1, Triton 3.1.0, mamba-ssm 2.2.4,
causal-conv1d 1.7.0).

Este módulo debe importarse ANTES de cualquier importación de ``hnet``. Al
importarse aplica los parches de forma idempotente y añade al ``sys.path`` el
directorio que contiene el paquete ``hnet``, cuyas importaciones internas son
absolutas (``from hnet.modules...``).

Parches aplicados:
    1. ``triton.set_allocator``: mamba-ssm lo invoca y Triton 3.1.0 no lo define.
    2. ``causal_conv1d_cuda.causal_conv1d_fwd`` (firma): mamba-ssm 2.2.4 llama con
       7 argumentos y espera recibir ``out``; causal-conv1d 1.7.0 exige 8
       argumentos (``out`` preasignado) y devuelve ``None``.
    3. ``causal_conv1d_cuda.causal_conv1d_fwd`` (alineación): ``xBC`` es un slice
       de ``zxbcdt`` cuyos strides en channel-last no son múltiplos de 8, lo que
       aborta el kernel. Se realiza una copia alineada; su coste debe tenerse en
       cuenta en los benchmarks.

El coste del parche 3 puede cuantificarse con ``SHIM_PROFILER``: desactivado por
defecto (sin coste adicional), al activarse contabiliza las copias de
alineación, los bytes copiados y, opcionalmente, su tiempo de GPU mediante
eventos CUDA, que no introducen sincronizaciones hasta la lectura del resumen.
Solo afecta al procesamiento de secuencias completas (prefill); la
decodificación incremental emplea ``causal_conv1d_update`` y no pasa por él.
"""

import os
import sys

import torch
import triton
import causal_conv1d_cuda

HNET_PARENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _dummy_set_allocator(allocator):
    """Sustituto vacío de ``triton.set_allocator`` para Triton 3.1.0."""
    pass


class ShimProfiler:
    """Contador del coste de las copias de alineación introducidas por el parche 3."""

    def __init__(self):
        self.enabled = False
        self.timing = False
        self.reset()

    def reset(self):
        self.fwd_calls = 0
        self.copies = 0
        self.bytes_copied = 0
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def enable(self, timing: bool = True):
        self.reset()
        self.enabled = True
        self.timing = timing

    def disable(self):
        self.enabled = False

    def start(self):
        """Registra el evento inicial de una posible copia (``None`` si no se mide el tiempo)."""
        if not self.timing:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def record_copy(self, tensor, start_event):
        """Contabiliza una copia del tamaño de ``tensor`` iniciada en ``start_event``."""
        self.copies += 1
        self.bytes_copied += tensor.numel() * tensor.element_size()
        if start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self._events.append((start_event, end_event))

    def summary(self) -> dict:
        """Devuelve los contadores acumulados; sincroniza con la GPU si se ha medido el tiempo."""
        time_ms = None
        if self.timing:
            if self._events:
                self._events[-1][1].synchronize()
            time_ms = sum(a.elapsed_time(b) for a, b in self._events)
        return {
            "fwd_calls": self.fwd_calls,
            "aligned_copies": self.copies,
            "bytes_copied": self.bytes_copied,
            "copy_time_ms": time_ms,
        }


SHIM_PROFILER = ShimProfiler()


def _align_profiled(x):
    """``_align_for_kernel`` con contabilidad opcional de la copia."""
    if not SHIM_PROFILER.enabled:
        return _align_for_kernel(x)
    start_event = SHIM_PROFILER.start()
    x_aligned = _align_for_kernel(x)
    if x_aligned is not x:
        SHIM_PROFILER.record_copy(x_aligned, start_event)
    return x_aligned


def _align_for_kernel(x):
    """Devuelve ``x`` con strides admitidos por el kernel (copia solo si es necesario)."""
    if x.stride(1) == 1 and (x.stride(0) % 8 != 0 or x.stride(2) % 8 != 0):
        if x.shape[1] % 8 == 0:
            return x.transpose(1, 2).contiguous().transpose(1, 2)
        return x.contiguous()
    if not x.is_contiguous() and x.stride(1) != 1:
        return x.contiguous()
    return x


def _make_patched_causal_conv1d_fwd(orig_fwd):
    """Construye el adaptador de ``causal_conv1d_fwd`` sobre la función original."""

    def _patched_causal_conv1d_fwd(*args):
        if SHIM_PROFILER.enabled:
            SHIM_PROFILER.fwd_calls += 1
        # Se distinguen las dos convenciones de llamada por el número de argumentos
        if len(args) == 8:
            return _current_causal_conv1d_fwd(*args)
        return _legacy_causal_conv1d_fwd(*args)

    def _current_causal_conv1d_fwd(
        x, weight, bias, seq_idx, initial_states, out, final_states_out, silu_activation
    ):
        # Convención nueva (8 argumentos, ``out`` preasignado por quien llama), empleada
        # por la propia interfaz de causal-conv1d (``causal_conv1d_fn``). Si ``x`` no está
        # alineado, se calcula sobre una copia alineada y se vuelca el resultado en ``out``.
        x_aligned = _align_profiled(x)
        if x_aligned is x:
            return orig_fwd(x, weight, bias, seq_idx, initial_states, out, final_states_out, silu_activation)
        out_aligned = torch.empty_like(x_aligned)
        orig_fwd(x_aligned, weight, bias, seq_idx, initial_states, out_aligned, final_states_out, silu_activation)
        start_event = SHIM_PROFILER.start() if SHIM_PROFILER.enabled else None
        out.copy_(out_aligned)
        if SHIM_PROFILER.enabled:
            SHIM_PROFILER.record_copy(out, start_event)

    def _legacy_causal_conv1d_fwd(
        x,
        weight,
        bias=None,
        seq_idx=None,
        initial_states=None,
        return_final_states=None,
        silu_activation=True,
    ):
        # Convención antigua (7 argumentos) empleada directamente por mamba-ssm 2.2.4
        x = _align_profiled(x)

        # Preasignación de las salidas requerida por la firma de 8 argumentos
        out = torch.empty_like(x)
        final_states_out = None
        if return_final_states:
            final_states_out = torch.empty(
                x.shape[0], x.shape[1], weight.shape[-1] - 1, device=x.device, dtype=x.dtype
            )

        orig_fwd(
            x,
            weight,
            bias,
            seq_idx,
            initial_states,
            out,
            final_states_out,
            bool(silu_activation)
        )

        return (out, final_states_out) if return_final_states else out

    _patched_causal_conv1d_fwd._hnet_compat_patched = True
    return _patched_causal_conv1d_fwd


def apply_patches():
    """Aplica todos los shims. Las llamadas repetidas no tienen efecto adicional."""
    if not hasattr(triton, "set_allocator"):
        triton.set_allocator = _dummy_set_allocator

    if not getattr(causal_conv1d_cuda.causal_conv1d_fwd, "_hnet_compat_patched", False):
        causal_conv1d_cuda.causal_conv1d_fwd = _make_patched_causal_conv1d_fwd(
            causal_conv1d_cuda.causal_conv1d_fwd
        )

    if HNET_PARENT_DIR not in sys.path:
        sys.path.append(HNET_PARENT_DIR)


apply_patches()
