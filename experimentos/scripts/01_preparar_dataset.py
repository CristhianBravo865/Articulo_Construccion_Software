"""Prepara los dos conjuntos de casos (ver DISENO.md, seccion 4.2).

E1: FixMe, hunks de Python validos (CVE reales). Muestra de n_casos con semilla fija.
E2: SecurityEval, funciones inseguras en las que Bandit reporta al menos un hallazgo.

Uso:
    python 01_preparar_dataset.py --fixme RUTA/FixMe-v1.db
Escribe datos/E1.jsonl, datos/E2.jsonl y datos/embudo.json

FixMe: https://zenodo.org/records/10955342 (CC-BY-4.0)
SecurityEval: https://github.com/s2e-lab/SecurityEval (se descarga, no se redistribuye)
"""
import argparse
import json
import random
import sqlite3
import urllib.request

from comun import RAIZ, bandit_lote, escribir_jsonl, es_valido, normalizar, cargar_config

URL_SECURITYEVAL = "https://raw.githubusercontent.com/s2e-lab/SecurityEval/main/dataset.jsonl"


def preparar_e1(db, cfg, embudo):
    con = sqlite3.connect(db)
    filas = con.execute(
        "SELECT h.rowid, h.file_id, h.file, h.code_before, h.code_after, p.url "
        "FROM hunk_collection h LEFT JOIN patch_collection p ON p.file_id = h.file_id "
        "WHERE h.programming_language = 'Python'"
    ).fetchall()
    embudo["E1_hunks_python"] = len({f[0] for f in filas})
    vistos, casos = set(), []
    for rowid, file_id, archivo, antes, despues, url in filas:
        if rowid in vistos:
            continue
        vistos.add(rowid)
        if not (es_valido(antes) and es_valido(despues)):
            continue
        if normalizar(antes) == normalizar(despues):
            continue
        casos.append({"id": f"E1-{rowid}", "archivo": archivo, "commit_url": url,
                      "vulnerable": normalizar(antes), "corregido": normalizar(despues)})
    embudo["E1_validos_y_distintos"] = len(casos)
    casos = [c for c in casos if len(c["vulnerable"]) <= cfg["max_chars_funcion"]]
    embudo["E1_longitud_ok"] = len(casos)
    random.Random(cfg["semilla"]).shuffle(casos)
    casos = sorted(casos[: cfg["n_casos"]], key=lambda c: c["id"])
    embudo["E1_muestra"] = len(casos)

    b_v = bandit_lote({c["id"]: c["vulnerable"] for c in casos})
    b_c = bandit_lote({c["id"]: c["corregido"] for c in casos})
    for c in casos:
        c["bandit_vulnerable"] = b_v[c["id"]] or []
        c["bandit_corregido"] = b_c[c["id"]] or []
        # Hallazgos que la correccion real elimina (solo informativo en E1)
        c["bandit_objetivo"] = sorted(set(c["bandit_vulnerable"]) - set(c["bandit_corregido"]))
    return casos


def preparar_e2(embudo):
    with urllib.request.urlopen(URL_SECURITYEVAL) as r:
        filas = [json.loads(l) for l in r.read().decode("utf-8").splitlines() if l.strip()]
    embudo["E2_securityeval_total"] = len(filas)
    b = bandit_lote({x["ID"]: x["Insecure_code"] for x in filas})
    casos = []
    for x in filas:
        ids = b[x["ID"]]
        if ids:
            casos.append({"id": "E2-" + x["ID"].removesuffix(".py"),
                          "cwe": x["ID"].split("_")[0],
                          "vulnerable": normalizar(x["Insecure_code"]),
                          "corregido": None,
                          "bandit_vulnerable": ids,
                          "bandit_objetivo": sorted(set(ids))})
    embudo["E2_con_hallazgo_bandit"] = len(casos)
    return casos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixme", required=True)
    args = ap.parse_args()
    cfg = cargar_config()
    embudo = {}
    e1 = preparar_e1(args.fixme, cfg, embudo)
    e2 = preparar_e2(embudo)
    escribir_jsonl(RAIZ / "datos" / "E1.jsonl", e1)
    escribir_jsonl(RAIZ / "datos" / "E2.jsonl", e2)
    (RAIZ / "datos" / "embudo.json").write_text(json.dumps(embudo, indent=2), encoding="utf-8")
    print(json.dumps(embudo, indent=2))


if __name__ == "__main__":
    main()
