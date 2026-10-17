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


def _make_patched_causal_conv1d_fwd(orig_fwd):
    """Construye el adaptador de ``causal_conv1d_fwd`` sobre la función original."""

    def _patched_causal_conv1d_fwd(
        x,
        weight,
        bias=None,
        seq_idx=None,
        initial_states=None,
        return_final_states=None,
        silu_activation=True,
        *args,
        **kwargs
    ):
        # Alineación de strides exigida por el kernel en formato channel-last
        if x.stride(1) == 1 and (x.stride(0) % 8 != 0 or x.stride(2) % 8 != 0):
            if x.shape[1] % 8 == 0:
                x = x.transpose(1, 2).contiguous().transpose(1, 2)
            else:
                x = x.contiguous()
        elif not x.is_contiguous() and x.stride(1) != 1:
            x = x.contiguous()

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
