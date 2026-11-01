"""Structure rules for the Financial Auditor's modules: run order, no LLM, cold imports."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

from app.agents.specialized.financial_auditor.modules import MODULES

BACKEND_ROOT = Path(__file__).resolve().parents[2]
MODULES_DIR = BACKEND_ROOT / "app/agents/specialized/financial_auditor/modules"
PACKAGE = "app.agents.specialized.financial_auditor.modules."
LLM_IMPORTS = ("app.agents.shared.llm_call", "app.agents.shared.prompts", "langchain", "langchain_openai", "openai")


def _imports(path: Path) -> set:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


class TestAuditModuleRules(unittest.TestCase):
    def test_modules_lists_every_module_folder(self):
        folders = {p.name for p in MODULES_DIR.iterdir() if p.is_dir() and not p.name.startswith("__")}
        self.assertEqual(folders, set(MODULES))

    def test_a_module_reads_only_modules_listed_before_it(self):
        for position, name in enumerate(MODULES):
            allowed = set(MODULES[: position + 1])
            for path in (MODULES_DIR / name).glob("*.py"):
                for imported in _imports(path):
                    if imported.startswith(PACKAGE):
                        other = imported[len(PACKAGE):].split(".")[0]
                        self.assertIn(other, allowed, f"{name}/{path.name} reads {other}, which runs later")

    def test_modules_never_call_the_llm(self):
        for name in MODULES:
            for path in (MODULES_DIR / name).glob("*.py"):
                for imported in _imports(path):
                    self.assertFalse(imported.startswith(LLM_IMPORTS), f"{name}/{path.name} imports {imported}")

    def test_each_module_imports_cold(self):
        for name in MODULES:
            done = subprocess.run(
                [sys.executable, "-c", f"import {PACKAGE}{name}"], cwd=BACKEND_ROOT, capture_output=True, text=True
            )
            self.assertEqual(done.returncode, 0, f"{name}: {done.stderr[-2000:]}")
