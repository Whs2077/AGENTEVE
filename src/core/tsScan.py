"""Lexical scan of TypeScript calls. The scan names functions; it does not build graph nodes."""

from pathlib import Path
import re

_TOKEN = re.compile(r"[A-Za-z_$][\w$]*|[{}()]|\.")
_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "function", "namespace", "async", "await"}

def typescript_calls(path):
    """Return stream, chat, and deepagent calls with the function that contains them."""
    functions, tokens = typescript_functions(Path(path))
    calls = []
    for index, (value, line, column) in enumerate(tokens):
        if not _is_ident(value) or index in functions["names"]:
            continue
        callee, end = _dotted(tokens, index)
        if end + 1 >= len(tokens) or tokens[end + 1][0] != "(":
            continue
        if callee.split(".")[-1] in _KEYWORDS:
            continue
        symbol = _stack_symbol(functions["ranges"], index) or "<module>"
        calls.append({"callee": callee, "symbol": symbol, "line": line, "column": column})
    return calls


def typescript_functions(path):
    tokens = _tokenize(mask_literals(path.read_text(encoding="utf-8")))
    name_indexes = set()
    ranges = []
    index = 0
    while index < len(tokens):
        value = tokens[index][0]
        if value in {"function", "namespace"} and _ident_at(tokens, index + 1):
            name_indexes.add(index + 1)
            open_at = _body_open(tokens, index + 2)
            if open_at is not None:
                ranges.append(_function_range(tokens, open_at, tokens[index + 1][0], tokens[index + 1][1]))
            index += 2
            continue
        if (value == "async" and _ident_at(tokens, index + 1) and _is(tokens, index + 2, "(")
                and not (index > 0 and tokens[index - 1][0] == "function")):
            name_indexes.add(index + 1)
            close = _match_paren(tokens, index + 2)
            open_at = None if close is None else _body_open(tokens, close + 1)
            if open_at is not None:
                ranges.append(_function_range(tokens, open_at, tokens[index + 1][0], tokens[index + 1][1]))
            index += 1
            continue
        index += 1
    for item in ranges:
        item["symbol"] = _stack_symbol(ranges, item["open"] + 1) or item["name"]
    return {"ranges": ranges, "names": name_indexes}, tokens


def _function_range(tokens, open_at, name, line):
    close = _match_brace(tokens, open_at)
    return {"name": name, "open": open_at, "close": len(tokens) if close is None else close, "line": line}


def _stack_symbol(functions, index):
    ranges = functions["ranges"] if isinstance(functions, dict) else functions
    active = [item for item in ranges if item["open"] < index < item["close"]]
    active.sort(key=lambda item: item["open"])
    if not active:
        return ""
    symbol = active[0]["name"]
    for item in active[1:]:
        symbol = f"{symbol}.{item['name']}"
    return symbol


def _body_open(tokens, start):
    index = start
    while index < len(tokens):
        value = tokens[index][0]
        if value == ";":
            return None
        if value == "{":
            close = _match_brace(tokens, index)
            if close is not None and close + 1 < len(tokens) and tokens[close + 1][0] == "{":
                index = close + 1
                continue
            return index
        if value == "(":
            close = _match_paren(tokens, index)
            if close is None:
                return None
            index = close + 1
            continue
        index += 1
    return None


def _match_paren(tokens, open_index):
    return _match(tokens, open_index, "(", ")")


def _match_brace(tokens, open_index):
    return _match(tokens, open_index, "{", "}")


def _match(tokens, open_index, opener, closer):
    depth = 0
    for index in range(open_index, len(tokens)):
        value = tokens[index][0]
        if value == opener:
            depth += 1
        elif value == closer:
            depth -= 1
            if depth == 0:
                return index
    return None


def _dotted(tokens, index):
    parts = [tokens[index][0]]
    cursor = index
    while cursor + 2 < len(tokens) and tokens[cursor + 1][0] == "." and _is_ident(tokens[cursor + 2][0]):
        parts.append(tokens[cursor + 2][0])
        cursor += 2
    return ".".join(parts), cursor


def _ident_at(tokens, index):
    return index < len(tokens) and _is_ident(tokens[index][0])


def _is(tokens, index, value):
    return index < len(tokens) and tokens[index][0] == value


def _is_ident(value):
    return value[:1].isalpha() or value[:1] in {"_", "$"}


def _tokenize(text):
    tokens = []
    index = 0
    line = 1
    column = 1
    while index < len(text):
        if text[index] == "\n":
            line += 1
            column = 1
            index += 1
            continue
        match = _TOKEN.match(text, index)
        if match and match.start() == index:
            tokens.append((match.group(), line, column))
            index = match.end()
            column += len(match.group())
            continue
        index += 1
        column += 1
    return tokens


def mask_literals(text):
    """Replace comments, strings, and regular expressions with spaces while keeping newlines."""
    blanks = []
    index = 0
    size = len(text)
    while index < size:
        char = text[index]
        nxt = text[index + 1] if index + 1 < size else ""
        if char == "/" and nxt == "/":
            end = text.find("\n", index)
            blanks.append((index, size if end < 0 else end))
            index = size if end < 0 else end
            continue
        if char == "/" and nxt == "*":
            end = text.find("*/", index + 2)
            stop = size if end < 0 else end + 2
            blanks.append((index, stop))
            index = stop
            continue
        if char in {"'", '"'}:
            index = _blank_quote(text, blanks, index, char)
            continue
        if char == "`":
            index = _blank_template(text, blanks, index)
            continue
        if char == "/" and _regex_before(text, index):
            end = _regex_end(text, index)
            if end is not None:
                blanks.append((index, end))
                index = end
                continue
        index += 1
    chars = list(text)
    for start, end in blanks:
        for cursor in range(start, end):
            if chars[cursor] != "\n":
                chars[cursor] = " "
    return "".join(chars)


def _regex_before(text, index):
    cursor = index - 1
    while cursor >= 0 and text[cursor] in " \t":
        cursor -= 1
    return cursor < 0 or text[cursor] in "={[(:,;!&|?+-*%~^<>\n"


def _regex_end(text, index):
    cursor = index + 1
    in_class = False
    while cursor < len(text) and text[cursor] != "\n":
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] == "[" and not in_class:
            in_class = True
        elif text[cursor] == "]" and in_class:
            in_class = False
        elif text[cursor] == "/" and not in_class:
            cursor += 1
            while cursor < len(text) and text[cursor].isalpha():
                cursor += 1
            return cursor
        cursor += 1
    return None


def _blank_quote(text, blanks, index, quote):
    cursor = index + 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] == "\n":
            break
        if text[cursor] == quote:
            blanks.append((index, cursor + 1))
            return cursor + 1
        cursor += 1
    blanks.append((index, cursor))
    return cursor


def _blank_template(text, blanks, index):
    cursor = index + 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] == "`":
            blanks.append((index, cursor + 1))
            return cursor + 1
        if text[cursor] == "$" and cursor + 1 < len(text) and text[cursor + 1] == "{":
            blanks.append((index, cursor))
            cursor = _skip_braced(text, blanks, cursor + 1)
            index = cursor
            continue
        cursor += 1
    blanks.append((index, len(text)))
    return len(text)


def _skip_braced(text, blanks, index):
    """Skip a `{...}` group. ``index`` points at the opening brace."""
    depth = 0
    cursor = index
    while cursor < len(text):
        if text[cursor] in {"'", '"'}:
            cursor = _blank_quote(text, blanks, cursor, text[cursor])
            continue
        if text[cursor] == "`":
            cursor = _blank_template(text, blanks, cursor)
            continue
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return cursor
