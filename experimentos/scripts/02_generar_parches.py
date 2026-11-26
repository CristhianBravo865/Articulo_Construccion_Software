"""Genera parches con una configuracion (modelo + bits) y registra tiempo y memoria.

Uso:
    python 02_generar_parches.py --experimento E2 --config Q-4 [--limite 10] [--hardware laptop]

Es reanudable: si el archivo de salida ya tiene casos, los salta (util en Colab).
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList)

from comun import (RAIZ, _recortar_prosa, bloques_cerrados, cargar_config, construir_entrada,
                   extraer_codigo, leer_jsonl)


class CortarTrasCodigo(StoppingCriteria):
    """Detiene la generacion cuando ya hay un bloque ```...``` cerrado con Python valido.

    Evita gastar tiempo en explicaciones; no cambia el codigo generado (decodificacion voraz).
    """

    def __init__(self, tok, n_prompt, cada=16):
        self.tok, self.n_prompt, self.cada, self.paso = tok, n_prompt, cada, 0

    def __call__(self, input_ids, scores, **kw):
        self.paso += 1
        if self.paso % self.cada:
            return False
        texto = self.tok.decode(input_ids[0][self.n_prompt:], skip_special_tokens=True)
        return any(_recortar_prosa(b) for b in bloques_cerrados(texto))


def cargar_modelo(model_id: str, bits: int):
    # dtype fp16 tambien en 8 y 4 bits: las capas no cuantizadas (embeddings, normas, lm_head)
    # quedarian en bfloat16 por defecto (config del modelo), que la T4 no soporta en hardware:
    # cada matmul se convierte bf16->fp16 y la generacion se vuelve muy lenta.
    kw = {"device_map": "auto", "dtype": torch.float16}
    if bits == 16:
        pass
    elif bits == 8:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif bits == 4:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        raise ValueError(f"bits no soportados por bitsandbytes: {bits}")
    tok = AutoTokenizer.from_pretrained(model_id)
    modelo = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    modelo.eval()
    return tok, modelo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experimento", required=True, choices=["E1", "E2"])
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--hardware", default="desconocido")
    args = ap.parse_args()

    cfg = cargar_config()
    c = cfg["configuraciones"][args.config]
    model_id = cfg["modelos"][c["modelo"]]
    plantilla = c.get("plantilla", "cruda")
    casos = leer_jsonl(RAIZ / "datos" / f"{args.experimento}.jsonl")[: args.limite]

    salida = RAIZ / "resultados" / f"gen_{args.experimento}_{args.config}_{args.hardware}.jsonl"
    hechos = {x["id"] for x in leer_jsonl(salida)} if salida.exists() else set()
    pendientes = [x for x in casos if x["id"] not in hechos]
    print(f"{args.config}: {model_id} a {c['bits']} bits, plantilla {plantilla} | {len(pendientes)} pendientes")
    if not pendientes:
        return

    torch.manual_seed(cfg["semilla"])
    t0 = time.perf_counter()
    tok, modelo = cargar_modelo(model_id, c["bits"])
    t_carga = time.perf_counter() - t0
    mem_modelo = torch.cuda.memory_allocated() / 2**20 if torch.cuda.is_available() else None
    print(f"carga {t_carga:.1f}s, memoria del modelo {mem_modelo} MiB")

    with open(salida, "a", encoding="utf-8") as f:
        for i, caso in enumerate(pendientes, 1):
            ent = construir_entrada(tok, caso["vulnerable"], plantilla).to(modelo.device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                out = modelo.generate(
                    **ent,
                    max_new_tokens=cfg["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    stopping_criteria=StoppingCriteriaList(
                        [CortarTrasCodigo(tok, ent["input_ids"].shape[1])]),
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t
            nuevos = out[0][ent["input_ids"].shape[1]:]
            respuesta = tok.decode(nuevos, skip_special_tokens=True)
            fila = {
                "id": caso["id"],
                "config": args.config,
                "modelo": model_id,
                "bits": c["bits"],
                "plantilla": plantilla,
                "hardware": args.hardware,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "tokens_entrada": int(ent["input_ids"].shape[1]),
                "tokens_generados": int(len(nuevos)),
                "segundos": round(dt, 3),
                "vram_pico_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if torch.cuda.is_available() else None,
                "vram_modelo_mib": round(mem_modelo, 1) if mem_modelo else None,
                "segundos_carga": round(t_carga, 1),
                "respuesta": respuesta,
                "parche": extraer_codigo(respuesta),
            }
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(pendientes)}] {caso['id']} {dt:.1f}s {len(nuevos)} tok")


if __name__ == "__main__":
    main()
