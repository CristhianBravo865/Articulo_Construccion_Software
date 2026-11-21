"""Tabla resumen por configuracion + McNemar y Jaccard contra la linea base FP16.

Uso:
    python 04_analizar.py           (lee resultados/eval_*.jsonl)
Escribe resultados/resumen.csv y resultados/comparaciones.csv
"""
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from comun import RAIZ, leer_jsonl

LINEA_BASE = {"L": "L-16", "Q": "Q-16"}


def main():
    filas = []
    for ruta in sorted((RAIZ / "resultados").glob("eval_*.jsonl")):
        filas += leer_jsonl(ruta)
    df = pd.DataFrame(filas)
    df["clave"] = df["config"] + "@" + df["hardware"]

    resumen = df.groupby("clave").agg(
        n=("id", "count"),
        valido=("valido", "mean"),
        eliminada=("eliminada", "mean"),
        introducida=("introducida", "mean"),
        reparacion_segura=("reparacion_segura", "mean"),
        identico=("identico_vulnerable", "mean"),
        codebleu=("codebleu", "mean"),
        rouge_l=("rouge_l", "mean"),
        seg_por_parche=("segundos", "mean"),
        tok_por_seg=("tokens_generados", "sum"),
        vram_pico_mib=("vram_pico_mib", "max"),
    )
    resumen["tok_por_seg"] = resumen["tok_por_seg"] / df.groupby("clave")["segundos"].sum()
    resumen = resumen.round(4)
    resumen.to_csv(RAIZ / "resultados" / "resumen.csv")
    print(resumen.to_string())

    comps = []
    for clave in resumen.index:
        config, hw = clave.split("@")
        base = LINEA_BASE[config[0]] + "@" + hw
        if clave == base or base not in resumen.index:
            continue
        a = df[df.clave == base].set_index("id")["reparacion_segura"]
        b = df[df.clave == clave].set_index("id")["reparacion_segura"]
        ids = a.index.intersection(b.index)
        a, b = a[ids], b[ids]
        tabla = [[int((a & b).sum()), int((a & ~b).sum())],
                 [int((~a & b).sum()), int((~a & ~b).sum())]]
        p = mcnemar(tabla, exact=True).pvalue
        ra, rb = set(ids[a.values]), set(ids[b.values])
        jac = len(ra & rb) / len(ra | rb) if ra | rb else float("nan")
        comps.append({"base": base, "config": clave, "n": len(ids),
                      "solo_base": tabla[0][1], "solo_cuantizado": tabla[1][0],
                      "mcnemar_p": round(p, 4), "jaccard": round(jac, 3)})
    if comps:
        c = pd.DataFrame(comps)
        c.to_csv(RAIZ / "resultados" / "comparaciones.csv", index=False)
        print("\n" + c.to_string(index=False))


if __name__ == "__main__":
    main()
