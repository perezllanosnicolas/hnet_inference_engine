import sys
import os
import torch

# Aplicación de los shims de compatibilidad antes de importar H-Net
sys.path.append(os.path.abspath("src/models/architectures"))
import hnet_compat  # noqa: F401

from hnet.models.config_hnet import HNetConfig, AttnConfig, SSMConfig
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.utils.tokenizers import ByteTokenizer


def test_hnet_forward():
    print("--- Prueba de humo de la arquitectura H-Net ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Dispositivo de ejecución: {device} | Precisión: {dtype}")
    
    config = HNetConfig(
        vocab_size=256,              
        d_model=[128, 256, 128],          
        d_intermediate=[256, 512, 256],   
        arch_layout=[
            "m1t1",     # Etapa 0: codificador (resolución de byte)
            ["M2T1"],   # Etapa 1: red principal (resolución de chunk)
            "m1t1"      # Etapa 0: decodificador (resolución de byte)
        ],
        attn_cfg=AttnConfig(
            num_heads=[4, 8, 4],
            rotary_emb_dim=[0, 0, 0],
            window_size=[-1, -1, -1]
        ),
        ssm_cfg=SSMConfig()
    )

    print("Instanciación del modelo (pesos aleatorios)...")
    model = HNetForCausalLM(config).to(device=device, dtype=dtype)
    model.eval() 
    
    tokenizer = ByteTokenizer()
    texto_prueba = "Hola URJC, probando inferencia en H-Net."
    tokens = tokenizer.encode([texto_prueba])
    
    input_ids = torch.tensor(tokens[0]["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"Texto original: '{texto_prueba}'")
    print(f"Bytes de entrada (forma): {input_ids.shape}")

    print("Ejecución del forward pass...")
    with torch.no_grad():
        output = model(input_ids)
    
    print("\nForward pass completado sin errores.")
    print(f"Forma de los logits de salida: {output.logits.shape}")
    
    if hasattr(output, "bpred_output") and output.bpred_output:
        router_output = output.bpred_output[0]
        print(f"Forma de la máscara de fronteras (boundary_mask): {router_output.boundary_mask.shape}")
        chunks_creados = router_output.boundary_mask.sum().item()
        print(f"Bytes originales: {input_ids.shape[1]} -> Chunks comprimidos: {chunks_creados}")

if __name__ == "__main__":
    test_hnet_forward()