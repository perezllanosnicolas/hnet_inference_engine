import sys
import os
import triton
import torch

# 1. Compatibilidad para Triton en PyTorch 2.5.1
if not hasattr(triton, 'set_allocator'):
    def _dummy_set_allocator(allocator):
        pass
    triton.set_allocator = _dummy_set_allocator

# 2. Adapter Shim para causal-conv1d 1.7.0 (alineación de strides y firma de 8 argumentos)
import causal_conv1d_cuda

_orig_causal_conv1d_fwd = causal_conv1d_cuda.causal_conv1d_fwd

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
    if x.stride(1) == 1 and (x.stride(0) % 8 != 0 or x.stride(2) % 8 != 0):
        if x.shape[1] % 8 == 0:
            x = x.transpose(1, 2).contiguous().transpose(1, 2)
        else:
            x = x.contiguous()
    elif not x.is_contiguous() and x.stride(1) != 1:
        x = x.contiguous()

    out = torch.empty_like(x)
    final_states_out = None
    if return_final_states:
        final_states_out = torch.empty(
            x.shape[0], x.shape[1], weight.shape[-1] - 1, device=x.device, dtype=x.dtype
        )

    _orig_causal_conv1d_fwd(
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

causal_conv1d_cuda.causal_conv1d_fwd = _patched_causal_conv1d_fwd

# 3. Inyección de rutas e importación de H-Net
sys.path.append(os.path.abspath("src/models/architectures"))

import hnet.modules.mha as hnet_mha
from hnet.models.config_hnet import HNetConfig, AttnConfig, SSMConfig
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.utils.tokenizers import ByteTokenizer

# 4. Adapter Shim para FlashAttention: asegurar que max_seqlen sea int nativo
_orig_flash_self_fwd = hnet_mha.FlashSelfAttention.forward

def _patched_flash_self_fwd(self, *args, **kwargs):
    args = list(args)
    if len(args) > 3 and isinstance(args[3], torch.Tensor):
        args[3] = int(args[3].item())
    if "max_seqlen" in kwargs and isinstance(kwargs["max_seqlen"], torch.Tensor):
        kwargs["max_seqlen"] = int(kwargs["max_seqlen"].item())
    return _orig_flash_self_fwd(self, *args, **kwargs)

hnet_mha.FlashSelfAttention.forward = _patched_flash_self_fwd

if hasattr(hnet_mha, "FlashCrossAttention"):
    _orig_flash_cross_fwd = hnet_mha.FlashCrossAttention.forward
    def _patched_flash_cross_fwd(self, *args, **kwargs):
        args = list(args)
        if len(args) > 4 and isinstance(args[4], torch.Tensor):
            args[4] = int(args[4].item())
        if len(args) > 6 and isinstance(args[6], torch.Tensor):
            args[6] = int(args[6].item())
        if "max_seqlen" in kwargs and isinstance(kwargs["max_seqlen"], torch.Tensor):
            kwargs["max_seqlen"] = int(kwargs["max_seqlen"].item())
        if "max_seqlen_k" in kwargs and isinstance(kwargs["max_seqlen_k"], torch.Tensor):
            kwargs["max_seqlen_k"] = int(kwargs["max_seqlen_k"].item())
        return _orig_flash_cross_fwd(self, *args, **kwargs)
    hnet_mha.FlashCrossAttention.forward = _patched_flash_cross_fwd


def test_hnet_forward():
    print("--- Iniciando prueba de humo de Arquitectura H-Net ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Dispositivo de ejecución: {device} | Precisión: {dtype}")
    
    config = HNetConfig(
        vocab_size=256,              
        d_model=[128, 256, 128],          
        d_intermediate=[256, 512, 256],   
        arch_layout=[
            "m1t1",     # Stage 0: Downsampler
            ["M2T1"],   # Stage 1: Backbone
            "m1t1"      # Stage 2: Upsampler
        ],
        attn_cfg=AttnConfig(
            num_heads=[4, 8, 4],
            rotary_emb_dim=[0, 0, 0],
            window_size=[(-1, -1), (-1, -1), (-1, -1)]
        ),
        ssm_cfg=SSMConfig()
    )

    print("Instanciando el modelo (pesos aleatorios)...")
    model = HNetForCausalLM(config).to(device=device, dtype=dtype)
    model.eval() 
    
    tokenizer = ByteTokenizer()
    texto_prueba = "Hola URJC, probando inferencia en H-Net."
    tokens = tokenizer.encode([texto_prueba])
    
    input_ids = torch.tensor(tokens[0]["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"Texto original: '{texto_prueba}'")
    print(f"Bytes de entrada (forma): {input_ids.shape}")

    print("Ejecutando Forward Pass en GPU...")
    with torch.no_grad():
        output = model(input_ids)
    
    print("\n✅ ¡ÉXITO TOTAL! El modelo ha procesado los bytes en la GPU correctamente.")
    print(f"Forma de los Logits de salida: {output.logits.shape}")
    
    if hasattr(output, "bpred_output") and output.bpred_output:
        router_output = output.bpred_output[0]
        print(f"Máscara de fronteras (Boundary Mask): {router_output.boundary_mask.shape}")
        chunks_creados = router_output.boundary_mask.sum().item()
        print(f"Bytes originales: {input_ids.shape[1]} -> Chunks comprimidos: {chunks_creados}")

if __name__ == "__main__":
    test_hnet_forward()