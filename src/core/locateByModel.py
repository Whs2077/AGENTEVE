"""Bucket 3: ask a model for one API location and accept only a checked file, line, and symbol."""

import ast
import json
from pathlib import Path

from src.core.apiSite import ApiBucket, ApiProtocol, iter_source_files, make_site
from src.core.graphModel import Unsupported
from src.core.pyScan import parents, symbol_covering

_RULE = ("Return one JSON object with file, line, and symbol. Return only the call location. "
         "Do not return graph nodes or edges.")


def choose_among(ask, sites):
    """Keep the located sites whose symbol is the one the model chose."""
    rows = "\n".join(f"{site.file}:{site.line} {site.symbol} {site.protocol}" for site in sites)
    choice = _model_choice(ask, "These source locations disagree about the large language model API. "
                                f"{_RULE}\n{rows}")
    chosen = [site for site in sites if _same_symbol(site, choice)]
    if not chosen:
        raise Unsupported("The model locator chose a symbol that is not one of the located API sites")
    return chosen


def locate_by_model(ask, root):
    """Ask for a location when no fixed bucket found one, then check that the symbol covers the line."""
    preview = []
    for file in list(iter_source_files(root))[:4]:
        lines = file.read_text(encoding="utf-8").splitlines()[:80]
        preview.append(f"## {file.name}\n" + "\n".join(lines))
    choice = _model_choice(ask, f"Locate one large language model API call. {_RULE}\n\n" + "\n\n".join(preview))
    file = _resolve_choice_file(root, choice["file"])
    line = int(choice["line"])
    text = file.read_text(encoding="utf-8")
    if file.suffix.lower() != ".py":
        if line < 1 or line > len(text.splitlines()):
            raise Unsupported(f"The model locator line is outside {file.name}")
        return make_site(file, line, 1, choice["symbol"], ApiProtocol.MODEL_LOCATED, ApiBucket.MODEL)
    tree = ast.parse(text, filename=str(file))
    symbol = symbol_covering(tree, parents(tree), line)
    wanted = str(choice["symbol"])
    if symbol != wanted and not symbol.endswith("." + wanted):
        raise Unsupported(f"The model locator symbol {wanted} does not cover line {line} of {file.name}")
    return make_site(file, line, 1, symbol, ApiProtocol.MODEL_LOCATED, ApiBucket.MODEL)


def _model_choice(ask, prompt):
    raw = ask(prompt)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as error:
            raise Unsupported("The model locator did not return a JSON object") from error
    if not isinstance(raw, dict) or not {"file", "line", "symbol"} <= set(raw):
        raise Unsupported("The model locator must return file, line, and symbol")
    return raw


def _same_symbol(site, choice):
    wanted = choice["symbol"]
    return site.file == Path(choice["file"]).name and (
        site.symbol == wanted or site.symbol.endswith(":" + wanted) or site.symbol.endswith("." + wanted))


def _resolve_choice_file(root, name):
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    matches = [file for file in iter_source_files(root) if file.name == candidate.name or file.as_posix().endswith(name)]
    if len(matches) != 1:
        raise Unsupported(f"The model locator file {name} is not one source file under {root}")
    return matches[0]
