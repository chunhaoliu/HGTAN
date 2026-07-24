"""Generate an additions-only highlighted manuscript from two LaTeX sources."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


DELETION_BLOCKS = (
    (r"\DIFdelbeginFL", r"\DIFdelendFL"),
    (r"\DIFdelbegin", r"\DIFdelend"),
)
ADDITION_MARKERS = (
    r"\DIFaddbeginFL",
    r"\DIFaddendFL",
    r"\DIFaddbegin",
    r"\DIFaddend",
)


def remove_braced_command(text: str, start: int, command: str) -> int:
    cursor = start + len(command)
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        return start + len(command)

    depth = 0
    escaped = False
    for cursor in range(cursor, len(text)):
        char = text[cursor]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cursor + 1
    raise ValueError(f"Unbalanced argument for {command} at offset {start}")


def clean_diff_body(body: str) -> str:
    output: list[str] = []
    cursor = 0
    while cursor < len(body):
        matched = False
        for begin, end in DELETION_BLOCKS:
            if body.startswith(begin, cursor):
                end_at = body.find(end, cursor + len(begin))
                if end_at < 0:
                    raise ValueError(f"Missing {end} for deletion block at offset {cursor}")
                cursor = end_at + len(end)
                matched = True
                break
        if matched:
            continue

        for command in (r"\DIFdelFL", r"\DIFdel"):
            if body.startswith(command, cursor):
                cursor = remove_braced_command(body, cursor, command)
                matched = True
                break
        if matched:
            continue

        marker = next((item for item in ADDITION_MARKERS if body.startswith(item, cursor)), None)
        if marker is not None:
            cursor += len(marker)
            continue

        output.append(body[cursor])
        cursor += 1

    cleaned = "".join(output)
    cleaned = re.sub(r"(?m)^[ \t]*%DIF(?:[^\r\n]*)(?:\r?\n|$)", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def use_additions_only_style(preamble: str) -> str:
    start = preamble.index("%DIF PREAMBLE EXTENSION ADDED BY LATEXDIFF")
    safe = preamble.index("%DIF SAFE PREAMBLE", start)
    replacement = (
        "%DIF PREAMBLE EXTENSION ADDED BY LATEXDIFF\n"
        r"\RequirePackage{color}\definecolor{RevisionBlue}{rgb}{0.00,0.32,0.61}"
        " %DIF PREAMBLE\n"
        r"\providecommand{\DIFadd}[1]{{\protect\color{RevisionBlue}#1}}"
        " %DIF PREAMBLE\n"
        r"\providecommand{\DIFdel}[1]{}"
        " %DIF PREAMBLE\n"
        r"\pdfstringdefDisableCommands{\def\DIFadd#1{#1}\def\DIFdel#1{}}"
        " %DIF PREAMBLE\n"
    )
    return preamble[:start] + replacement + preamble[safe:]


def generate_highlighted(old_source: Path, new_source: Path, output: Path) -> None:
    result = subprocess.run(
        [
            "latexdiff",
            "--type=CFONT",
            "--no-del",
            "--graphics-markup=none",
            "--encoding=utf8",
            str(old_source),
            str(new_source),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    document_marker = r"\begin{document}"
    preamble, body = result.stdout.split(document_marker, maxsplit=1)
    body = re.sub(r"(?m)^[ \t]*%DIF(?:[^\r\n]*)(?:\r?\n|$)", "", body)
    highlighted = (
        use_additions_only_style(preamble)
        + document_marker
        + body
    )

    rendered_body = highlighted.split(document_marker, maxsplit=1)[1]
    if r"\DIFdel" in rendered_body:
        raise ValueError("Deletion markup remains in the highlighted manuscript body")
    if r"\DIFadd{" not in rendered_body:
        raise ValueError("No revision additions were detected")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(highlighted, encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_highlighted(args.old.resolve(), args.new.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
