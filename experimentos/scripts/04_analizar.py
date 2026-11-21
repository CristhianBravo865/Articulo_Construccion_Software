"""Tabla resumen por experimento/configuracion + McNemar y Jaccard contra la linea base FP16.

Uso:
    python 04_analizar.py           (lee resultados/eval_*.jsonl)
Escribe resultados/resumen.csv y resultados/comparaciones.csv

Variable binaria comparada con McNemar:
  E2 -> reparacion_segura (oraculo Bandit: eliminada, sin introducir, cambio valido)
  E1 -> introducida (el parche agrega hallazgos de Bandit que no tenia el codigo vulnerable)
"""
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from comun import RAIZ, leer_jsonl

LINEA_BASE = {"L": "L-16", "Q": "Q-16"}
VARIABLE = {"E1": "introducida", "E2": "reparacion_segura"}
BOOL = ["valido", "cambio_valido", "identico_vulnerable", "eliminada", "introducida",
        "reparacion_segura"]


def main():
    filas = []
    for ruta in sorted((RAIZ / "resultados").glob("eval_*.jsonl")):
        filas += leer_jsonl(ruta)
    df = pd.DataFrame(filas)
    for col in BOOL + ["codebleu", "rouge_l"]:
        df[col] = pd.to_numeric(df[col].map(lambda v: np.nan if v is None else float(v)))
    df["grupo"] = df["experimento"] + "|" + df["config"] + "|" + df["hardware"]

    g = df.groupby("grupo")
    resumen = g[BOOL + ["codebleu", "rouge_l", "segundos"]].mean()
    resumen.insert(0, "n", g["id"].count())
    resumen["tok_por_seg"] = g["tokens_generados"].sum() / g["segundos"].sum()
    resumen["vram_pico_mib"] = g["vram_pico_mib"].max()
    resumen = resumen.round(4)
    resumen.to_csv(RAIZ / "resultados" / "resumen.csv")
    print(resumen.to_string())

    comps = []
    for grupo in resumen.index:
        exp, config, hw = grupo.split("|")
        base = f"{exp}|{LINEA_BASE[config[0]]}|{hw}"
        if grupo == base or base not in resumen.index:
            continue
        var = VARIABLE[exp]
        a = df[df.grupo == base].set_index("id")[var].dropna().astype(bool)
        b = df[df.grupo == grupo].set_index("id")[var].dropna().astype(bool)
        ids = a.index.intersection(b.index)
        a, b = a[ids], b[ids]
        tabla = [[int((a & b).sum()), int((a & ~b).sum())],
                 [int((~a & b).sum()), int((~a & ~b).sum())]]
        p = mcnemar(tabla, exact=True).pvalue
        ra, rb = set(ids[a.values]), set(ids[b.values])
        jac = len(ra & rb) / len(ra | rb) if ra | rb else float("nan")
        comps.append({"experimento": exp, "base": base, "comparada": grupo, "variable": var,
                      "n": len(ids), "tasa_base": round(a.mean(), 4),
                      "tasa_comparada": round(b.mean(), 4),
                      "solo_base": tabla[0][1], "solo_comparada": tabla[1][0],
                      "mcnemar_p": round(p, 4), "jaccard": round(jac, 3)})
    if comps:
        c = pd.DataFrame(comps)
        c.to_csv(RAIZ / "resultados" / "comparaciones.csv", index=False)
        print("\n" + c.to_string(index=False))


if __name__ == "__main__":
    main()
