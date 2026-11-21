"""Utilidades compartidas: prompt de PatchLM, extraccion de codigo y Bandit."""
import ast
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def cargar_config():
    import yaml
    return yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))


# Prompt de evaluacion copiado de PatchLM (source/prompt.py, generate_eval_prompt_codellama).
# Se respeta el texto original, incluida la frase "its vulnerable".
PROMPT_PATCHLM = (
    "You are a powerful code-fixing model.\n"
    "Your job is to analyze and fix vulnerabilities in code.\n"
    "You are given a snippet of vulnerable code and its vulnerable.\n"
    "You must output the fixed version of the code snippet.\n\n"
    "### Input:\n{question}\n\n"
    "### vulnerable:\n{vulnerable}\n\n"
    "### Response:\n"
)
PREGUNTA = "What is the fix version of the code for the following vulnerability?"


def construir_prompt(codigo_vulnerable: str) -> str:
    return PROMPT_PATCHLM.format(question=PREGUNTA, vulnerable=codigo_vulnerable)


def extraer_codigo(respuesta: str) -> str:
    """Toma el primer bloque ```...``` si existe; si no, el texto completo."""
    m = re.search(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)(```|$)", respuesta, re.S)
    codigo = m.group(1) if m else respuesta
    # Corta si el modelo empieza otra seccion del prompt
    codigo = re.split(r"\n### ", codigo)[0]
    return codigo.strip("\n")


def normalizar(codigo: str) -> str:
    return textwrap.dedent(codigo).strip("\n") + "\n"


def es_valido(codigo: str) -> bool:
    try:
        ast.parse(normalizar(codigo))
        return True
    except (SyntaxError, ValueError):
        return False


def bandit_ids(codigo: str) -> list[str] | None:
    """Devuelve la lista de test_id de Bandit (p. ej. B602) o None si no se pudo analizar."""
    if not es_valido(codigo):
        return None
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "fragmento.py"
        f.write_text(normalizar(codigo), encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-m", "bandit", "-q", "-f", "json", str(f)],
            capture_output=True, text=True, encoding="utf-8",
        )
        try:
            datos = json.loads(r.stdout)
        except json.JSONDecodeError:
            return None
        if datos.get("errors"):
            return None
        return sorted(x["test_id"] for x in datos["results"])


def bandit_lote(codigos: dict[str, str]) -> dict[str, list[str] | None]:
    """Como bandit_ids pero para muchos fragmentos en una sola llamada a Bandit."""
    res = {k: None for k in codigos}
    with tempfile.TemporaryDirectory() as d:
        nombres = {}
        for i, (k, cod) in enumerate(codigos.items()):
            if es_valido(cod):
                f = Path(d) / f"f{i}.py"
                f.write_text(normalizar(cod), encoding="utf-8")
                nombres[f.stem] = k
                res[k] = []
        if not nombres:
            return res
        r = subprocess.run(
            [sys.executable, "-m", "bandit", "-q", "-r", "-f", "json", d],
            capture_output=True, text=True, encoding="utf-8",
        )
        datos = json.loads(r.stdout)
        for e in datos.get("errors", []):
            res[nombres[Path(e["filename"]).stem]] = None
        for x in datos["results"]:
            k = nombres[Path(x["filename"]).stem]
            if res[k] is not None:
                res[k].append(x["test_id"])
    return {k: (sorted(v) if v is not None else None) for k, v in res.items()}


def leer_jsonl(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def escribir_jsonl(ruta, filas):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        for x in filas:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
