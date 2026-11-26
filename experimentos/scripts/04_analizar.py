"""Tabla resumen por experimento/configuracion + McNemar y Jaccard contra la linea base FP16.

Uso:
    python 04_analizar.py           (lee resultados/eval_*.jsonl)
Escribe resultados/resumen.csv y resultados/comparaciones.csv

Comparaciones contra la linea base FP16 del mismo modelo y hardware:
  Binarias (McNemar exacta): E2 reparacion_segura, cambio_valido, introducida;
                             E1 cambio_valido, introducida.
  Continuas (Wilcoxon pareada): E1 codebleu, rouge_l.
  "solo_base"/"solo_comparada": casos en que solo una de las dos acierta (o es mayor).
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from comun import RAIZ, leer_jsonl

BINARIAS = {"E1": ["cambio_valido", "introducida"],
            "E2": ["reparacion_segura", "reparacion_segura_sin_imports", "cambio_valido",
                   "introducida"]}
CONTINUAS = {"E1": ["codebleu", "rouge_l"], "E2": []}
BOOL = ["valido", "cambio_valido", "identico_vulnerable", "eliminada", "introducida",
        "eliminada_sin_imports", "reparacion_segura_sin_imports",
        "reparacion_segura"]


def main():
    filas = []
    for ruta in sorted((RAIZ / "resultados").glob("eval_*.jsonl")):
        filas += leer_jsonl(ruta)
    df = pd.DataFrame(filas)
    # En E1 hay hunks cuya version vulnerable esta vacia: la correccion crea el archivo entero,
    # asi que no hay codigo que reparar. Se excluyen en todas las configuraciones por igual.
    e1 = leer_jsonl(RAIZ / "datos" / "E1.jsonl")
    excluir = {c["id"] for c in e1 if not c["vulnerable"].strip()}
    df = df[~df["id"].isin(excluir)].copy()
    print(f"E1: {len(excluir)} casos excluidos por codigo vulnerable vacio -> n = {len(e1) - len(excluir)}")
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
        base = f"{exp}|{config.split('-')[0]}-16|{hw}"
        if grupo == base or base not in resumen.index:
            continue
        da = df[df.grupo == base].set_index("id")
        db = df[df.grupo == grupo].set_index("id")
        for var in BINARIAS[exp]:
            a, b = da[var].dropna().astype(bool), db[var].dropna().astype(bool)
            ids = a.index.intersection(b.index)
            a, b = a[ids], b[ids]
            tabla = [[int((a & b).sum()), int((a & ~b).sum())],
                     [int((~a & b).sum()), int((~a & ~b).sum())]]
            p = mcnemar(tabla, exact=True).pvalue
            ra, rb = set(ids[a.values]), set(ids[b.values])
            jac = len(ra & rb) / len(ra | rb) if ra | rb else float("nan")
            comps.append({"experimento": exp, "base": base, "comparada": grupo,
                          "variable": var, "prueba": "McNemar exacta", "n": len(ids),
                          "valor_base": round(a.mean(), 4), "valor_comparada": round(b.mean(), 4),
                          "solo_base": tabla[0][1], "solo_comparada": tabla[1][0],
                          "p": round(p, 4), "jaccard": round(jac, 3)})
        for var in CONTINUAS[exp]:
            a, b = da[var].dropna(), db[var].dropna()
            ids = a.index.intersection(b.index)
            a, b = a[ids], b[ids]
            p = wilcoxon(a, b).pvalue if (a != b).any() else 1.0
            comps.append({"experimento": exp, "base": base, "comparada": grupo,
                          "variable": var, "prueba": "Wilcoxon pareada", "n": len(ids),
                          "valor_base": round(a.mean(), 4), "valor_comparada": round(b.mean(), 4),
                          "solo_base": int((a > b).sum()), "solo_comparada": int((b > a).sum()),
                          "p": round(p, 4), "jaccard": None})
    if comps:
        c = pd.DataFrame(comps)
        c.to_csv(RAIZ / "resultados" / "comparaciones.csv", index=False)
        print("\n" + c.to_string(index=False))


if __name__ == "__main__":
    main()
