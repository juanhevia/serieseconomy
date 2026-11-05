"""Ejecuta el cuaderno local y exporta una vista HTML sin descargar datos."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
# Usar las fuentes incluidas con Matplotlib evita escanear todas las del sistema.
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
os.environ.setdefault("IPYTHONDIR", str(CACHE / "ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(CACHE / "jupyter"))

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec
from nbclient import NotebookClient
from nbconvert import HTMLExporter


class ProjectKernelManager(KernelManager):
    """Usa el Python del comando, sin registrar un kernel global."""

    @property
    def kernel_spec(self):
        return KernelSpec(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name="Python del proyecto",
            language="python",
        )


def main():
    path = ROOT / "notebooks/01_exploracion.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=180,
        resources={"metadata": {"path": str(ROOT)}},
        kernel_manager_class=ProjectKernelManager,
    )
    client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input = True
    html, _ = exporter.from_notebook_node(notebook)
    output = ROOT / "reports/01_exploracion.html"
    output.parent.mkdir(exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Cuaderno ejecutado: {path.relative_to(ROOT)}")
    print(f"Informe HTML: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
