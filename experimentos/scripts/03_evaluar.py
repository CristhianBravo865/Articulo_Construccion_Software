"""Evalua los parches generados: validez, Bandit (eliminada / introducida), CodeBLEU, ROUGE-L.

Uso:
    python 03_evaluar.py resultados/gen_Q-4_laptop-rtx3050.jsonl [...]
Escribe resultados/eval_<mismo nombre>.jsonl
"""
import sys
from pathlib import Path

from comun import RAIZ, bandit_ids, es_valido, escribir_jsonl, leer_jsonl, normalizar


def codebleu(ref: str, hip: str) -> float | None:
    try:
        from codebleu import calc_codebleu
        return round(calc_codebleu([ref], [hip], lang="python",
                                   weights=(0.25, 0.25, 0.25, 0.25))["codebleu"], 4)
    except Exception:
        return None


def rouge_l(ref: str, hip: str) -> float:
    from rouge_score import rouge_scorer
    s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return round(s.score(ref, hip)["rougeL"].fmeasure, 4)


def evaluar(gen, caso):
    parche = gen["parche"]
    vuln_ids = set(caso["bandit_vulnerable"])
    valido = es_valido(parche)
    ids = bandit_ids(parche) if valido else None
    identico = normalizar(parche).strip() == normalizar(caso["vulnerable"]).strip()
    vacio = len(parche.strip()) == 0
    eliminada = ids is not None and not (set(caso["bandit_objetivo"]) & set(ids))
    introducida = ids is not None and bool(set(ids) - vuln_ids)
    return {
        "id": gen["id"],
        "config": gen["config"],
        "hardware": gen["hardware"],
        "valido": valido,
        "vacio": vacio,
        "identico_vulnerable": identico,
        "bandit_parche": ids,
        "eliminada": eliminada,
        "introducida": introducida,
        "reparacion_segura": valido and eliminada and not introducida and not identico and not vacio,
        "codebleu": codebleu(caso["corregido"], parche) if not vacio else 0.0,
        "rouge_l": rouge_l(caso["corregido"], parche) if not vacio else 0.0,
        "segundos": gen["segundos"],
        "tokens_generados": gen["tokens_generados"],
        "vram_pico_mib": gen["vram_pico_mib"],
    }


def main():
    casos = {x["id"]: x for x in leer_jsonl(RAIZ / "datos" / "dataset.jsonl")}
    for ruta in sys.argv[1:]:
        ruta = Path(ruta)
        gens = leer_jsonl(ruta)
        filas = [evaluar(g, casos[g["id"]]) for g in gens]
        salida = ruta.with_name(ruta.name.replace("gen_", "eval_", 1))
        escribir_jsonl(salida, filas)
        n = len(filas)
        seg = sum(f["reparacion_segura"] for f in filas)
        print(f"{ruta.name}: {n} casos | reparacion segura {seg}/{n} = {seg / max(n, 1):.1%}")


if __name__ == "__main__":
    main()
