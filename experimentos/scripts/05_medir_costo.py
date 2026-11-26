"""Medicion controlada de costo (RQ3): mismos prompts, mismo numero de tokens, repeticiones.

En la corrida principal el tiempo por parche depende de cuanto escribe cada modelo y de
la carga de la maquina. Aqui se fija todo: 10 prompts (5 de E1 y 5 de E2), exactamente
N tokens generados por prompt, 1 calentamiento y R repeticiones.

Uso:
    python 05_medir_costo.py --config Q-4 --hardware laptop [--tokens 128] [--reps 3]
Escribe resultados/costo_<config>_<hardware>.json
"""
import argparse
import json
import statistics
import time

import torch

from comun import RAIZ, cargar_config, construir_entrada, leer_jsonl
from importlib import import_module

cargar_modelo = import_module("02_generar_parches").cargar_modelo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    cfg = cargar_config()
    c = cfg["configuraciones"][args.config]
    model_id = cfg["modelos"][c["modelo"]]
    plantilla = c.get("plantilla", "cruda")
    prompts = [x["vulnerable"]
               for exp in ("E1", "E2") for x in leer_jsonl(RAIZ / "datos" / f"{exp}.jsonl")[:5]]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tok, modelo = cargar_modelo(model_id, c["bits"])
    t_carga = time.perf_counter() - t0
    mem_modelo = torch.cuda.memory_allocated() / 2**20

    def generar(p):
        ent = construir_entrada(tok, p, plantilla).to(modelo.device)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            modelo.generate(**ent, max_new_tokens=args.tokens, min_new_tokens=args.tokens,
                            do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        return time.perf_counter() - t

    generar(prompts[0])  # calentamiento
    torch.cuda.reset_peak_memory_stats()
    tiempos = [[generar(p) for p in prompts] for _ in range(args.reps)]
    por_rep = [sum(r) / (len(prompts) * args.tokens) * 1000 for r in tiempos]  # ms/token

    res = {
        "config": args.config, "modelo": model_id, "bits": c["bits"], "plantilla": plantilla,
        "hardware": args.hardware, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "prompts": len(prompts), "tokens_por_prompt": args.tokens,
        "repeticiones": args.reps, "segundos_carga": round(t_carga, 1),
        "memoria_modelo_mib": round(mem_modelo, 1),
        "vram_pico_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "ms_por_token_media": round(statistics.mean(por_rep), 2),
        "ms_por_token_desv": round(statistics.stdev(por_rep), 2) if args.reps > 1 else None,
        "tokens_por_seg": round(1000 / statistics.mean(por_rep), 2),
        "ms_por_token_reps": [round(x, 2) for x in por_rep],
    }
    salida = RAIZ / "resultados" / f"costo_{args.config}_{args.hardware}.json"
    salida.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
