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


def _recortar_prosa(bloque: str) -> str | None:
    """Quita lineas iniciales de prosa hasta que el resto sea Python con al menos una
    sentencia (un bloque con solo comentarios no cuenta)."""
    lineas = bloque.strip("\n").splitlines()
    for i in range(min(len(lineas), 6)):
        resto = "\n".join(lineas[i:]).strip("\n")
        if not resto.strip():
            continue
        try:
            if ast.parse(normalizar(resto)).body:
                return resto
        except (SyntaxError, ValueError):
            pass
    return None


_FENCE = re.compile(r"^\s*```.*$", re.M)


def segmentos(texto: str, incluir_abierto: bool = False) -> list[str]:
    """Texto entre cada par de lineas ``` consecutivas.

    No asume que las cercas alternen abrir/cerrar: los modelos a veces abren un bloque,
    escriben prosa y abren otro. Probar todos los tramos consecutivos resuelve esos casos.
    """
    cortes = [m for m in _FENCE.finditer(texto)]
    segs = [texto[a.end():b.start()] for a, b in zip(cortes, cortes[1:])]
    if incluir_abierto and cortes:
        segs.append(texto[cortes[-1].end():])
    return segs


def bloques_cerrados(texto: str) -> list[str]:
    return segmentos(texto)


def extraer_codigo(respuesta: str) -> str:
    """Devuelve el primer tramo entre ``` que (quitando prosa inicial) sea Python valido.

    Si ninguno es valido, devuelve el primer tramo (o el texto) tal cual, para que la
    evaluacion lo cuente como parche invalido.
    """
    texto = re.split(r"\n### ", respuesta)[0]
    candidatos = segmentos(respuesta, incluir_abierto=True)
    for b in candidatos:
        limpio = _recortar_prosa(b)
        if limpio:
            return limpio
    if candidatos:
        return candidatos[0].strip("\n")
    return texto.strip("\n")


def normalizar(codigo: str) -> str:
    return textwrap.dedent(codigo).strip("\n") + "\n"


def es_valido(codigo: str) -> bool:
    try:
        ast.parse(normalizar(codigo))
        return True
    except (SyntaxError, ValueError):
        return False


def nombres_def(codigo: str) -> set[str]:
    """Nombres de funciones y clases definidas en el fragmento (vacio si no parsea)."""
    try:
        arbol = ast.parse(normalizar(codigo))
    except (SyntaxError, ValueError):
        return set()
    return {n.name for n in ast.walk(arbol)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


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
        if not r.stdout.strip():
            raise RuntimeError("Bandit no devolvio nada (esta instalado?). stderr: " + r.stderr[-2000:])
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
        if not r.stdout.strip():
            raise RuntimeError("Bandit no devolvio nada (esta instalado?). stderr: " + r.stderr[-2000:])
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
