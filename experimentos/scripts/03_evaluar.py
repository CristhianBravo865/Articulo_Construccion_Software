"""Evalua los parches generados: validez, Bandit (eliminada / introducida), CodeBLEU, ROUGE-L.

Uso:
    python 03_evaluar.py resultados/gen_Q-4_laptop-rtx3050.jsonl [...]
Escribe resultados/eval_<mismo nombre>.jsonl
"""
import sys
from pathlib import Path

from comun import RAIZ, bandit_ids, es_valido, escribir_jsonl, leer_jsonl, nombres_def, normalizar


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
    exp = gen["id"].split("-")[0]  # E1 o E2
    vuln_ids = set(caso["bandit_vulnerable"])
    valido = es_valido(parche)
    ids = bandit_ids(parche) if valido else None
    identico = normalizar(parche).strip() == normalizar(caso["vulnerable"]).strip()
    vacio = len(parche.strip()) == 0
    introducida = ids is not None and bool(set(ids) - vuln_ids)
    # El parche debe conservar las funciones/clases del original (evita "reparar" borrando)
    conserva = nombres_def(caso["vulnerable"]) <= nombres_def(parche)
    cambio_valido = valido and not identico and not vacio and conserva
    if caso["bandit_objetivo"]:
        # Hay oraculo: los hallazgos objetivo deben desaparecer
        eliminada = ids is not None and not (set(caso["bandit_objetivo"]) & set(ids))
        segura = cambio_valido and eliminada and not introducida
    else:
        eliminada = segura = None  # sin oraculo de eliminacion (la mayoria de E1)
    ref = caso.get("corregido")
    return {
        "id": gen["id"],
        "experimento": exp,
        "config": gen["config"],
        "hardware": gen["hardware"],
        "valido": valido,
        "vacio": vacio,
        "identico_vulnerable": identico,
        "conserva_funciones": conserva,
        "cambio_valido": cambio_valido,
        "bandit_parche": ids,
        "eliminada": eliminada,
        "introducida": introducida,
        "reparacion_segura": segura,
        "codebleu": (codebleu(ref, parche) if not vacio else 0.0) if ref else None,
        "rouge_l": (rouge_l(ref, parche) if not vacio else 0.0) if ref else None,
        "segundos": gen["segundos"],
        "tokens_generados": gen["tokens_generados"],
        "vram_pico_mib": gen["vram_pico_mib"],
    }


def main():
    casos = {}
    for nombre in ("E1", "E2"):
        ruta = RAIZ / "datos" / f"{nombre}.jsonl"
        if ruta.exists():
            casos.update({x["id"]: x for x in leer_jsonl(ruta)})
    for ruta in sys.argv[1:]:
        ruta = Path(ruta)
        gens = leer_jsonl(ruta)
        filas = [evaluar(g, casos[g["id"]]) for g in gens]
        salida = ruta.with_name(ruta.name.replace("gen_", "eval_", 1))
        escribir_jsonl(salida, filas)
        n = len(filas)
        cv = sum(f["cambio_valido"] for f in filas)
        intro = sum(f["introducida"] for f in filas)
        con_or = [f for f in filas if f["reparacion_segura"] is not None]
        seg = sum(f["reparacion_segura"] for f in con_or)
        print(f"{ruta.name}: {n} casos | cambio valido {cv} | introduce vuln {intro} | "
              f"reparacion segura {seg}/{len(con_or)}")


if __name__ == "__main__":
    main()
