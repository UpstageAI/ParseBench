"""Semantic normalization for Upstage Document Parse elements.

The API already returns authored Markdown and HTML.  These helpers keep that
content intact while making relationships that are explicit in the response
visible to ParseBench's Markdown/HTML consumers.
"""

import html
import json
import math
import re
from typing import Any

from bs4 import BeautifulSoup

_HEADING_SIZE_GROUP_RATIO = 0.97


def element_text(element: dict[str, Any]) -> str:
    content = element.get("content") or {}
    return str(content.get("text") or content.get("markdown") or "")


def element_html(element: dict[str, Any]) -> str:
    content = element.get("content") or {}
    return str(content.get("html") or content.get("markdown") or html.escape(element_text(element)))


def _semantic_table_headers(source_html: str) -> str:
    """Convert API ``scope`` annotations into standard HTML header cells."""
    if not source_html or "scope=" not in source_html.lower():
        return source_html

    soup = BeautifulSoup(source_html, "html.parser")
    changed = False
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row_index, row in enumerate(rows):
            cells = row.find_all(["td", "th"], recursive=False)
            for cell_index, cell in enumerate(cells):
                if cell.name != "td":
                    continue
                scope = str(cell.get("scope", "")).lower()
                if scope not in {"col", "colgroup", "row", "rowgroup"}:
                    continue
                cell.name = "th"
                if scope in {"col", "colgroup"} and row_index > 0 and cell_index == 0:
                    cell["scope"] = "row"
                changed = True
    return str(soup) if changed else source_html


def _element_plain_text(element: dict[str, Any]) -> str:
    text = element_text(element)
    if text:
        return " ".join(text.split())
    return BeautifulSoup(element_html(element), "html.parser").get_text(" ", strip=True)


def _nearby_chart_title(elements: list[dict[str, Any]], chart_index: int) -> str:
    """Return API-authored figure captions, with adjacent headings as fallback."""
    chart_page = elements[chart_index].get("page", 1)
    captions: list[str] = []
    headings: list[str] = []
    for candidate_index in range(max(0, chart_index - 4), min(len(elements), chart_index + 5)):
        if candidate_index == chart_index:
            continue
        candidate = elements[candidate_index]
        if candidate.get("page", 1) != chart_page:
            continue
        category = str(candidate.get("category") or "").lower()
        text = _element_plain_text(candidate)
        if not text:
            continue
        if category == "caption" and not re.match(r"^table\b", text, flags=re.IGNORECASE):
            captions.append(text)
        elif category.startswith("heading") and abs(candidate_index - chart_index) <= 2:
            headings.append(text)

    selected = captions or headings
    return " | ".join(dict.fromkeys(selected))


def _chart_ocr_residue(table: Any, figure: Any) -> str:
    """Return chart OCR labels that are not already represented by table cells."""
    ocr = figure.select_one(".chart-ocr-text")
    if ocr is None:
        return ""
    cell_words = {
        word.casefold()
        for cell in table.find_all(["th", "td"])
        for word in re.findall(r"[\w]+", cell.get_text(" ", strip=True), flags=re.UNICODE)
    }
    residue: list[str] = []
    for line in ocr.get_text("\n").splitlines():
        words = [
            word
            for word in re.findall(r"[\w]+(?:[.\-/][\w]+)*", line, flags=re.UNICODE)
            if word.casefold() not in cell_words
        ]
        candidate = " ".join(words).strip()
        if len(re.sub(r"[^A-Za-z]", "", candidate)) >= 2 and candidate not in residue:
            residue.append(candidate)
    return " | ".join(residue)


def _normalize_chart(source_html: str, external_title: str) -> str:
    """Bind the API's chart title, description, and residual labels to its table."""
    source_html = _semantic_table_headers(source_html)
    if not source_html or "<table" not in source_html.lower():
        return source_html

    soup = BeautifulSoup(source_html, "html.parser")
    for table in soup.find_all("table"):
        if table.find("caption", recursive=False) is not None:
            continue
        figure = table.find_parent("figure") or soup
        description_node = figure.select_one(".chart-description")
        description = description_node.get_text(" ", strip=True) if description_node else ""
        context = " | ".join(part for part in (external_title, _chart_ocr_residue(table, figure), description) if part)
        if context:
            caption = soup.new_tag("caption")
            caption.string = context
            table.insert(0, caption)
    return str(soup)


def _font_size_proxy(element: dict[str, Any]) -> float:
    coordinates = element.get("coordinates") or []
    if len(coordinates) < 4:
        return 0.0
    try:
        xs = [float(point["x"]) for point in coordinates]
        ys = [float(point["y"]) for point in coordinates]
    except (KeyError, TypeError, ValueError):
        return 0.0
    area = max(0.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    character_count = max(1, len(re.sub(r"\s+", "", element_text(element))))
    return math.sqrt(area / character_count)


def _heading_levels(elements: list[dict[str, Any]]) -> dict[int, int]:
    headings = [
        (index, element)
        for index, element in enumerate(elements)
        if str(element.get("category") or "").lower().startswith("heading")
    ]
    authored_levels: set[int] = set()
    for _, element in headings:
        match = re.match(r"^\s*(#{1,6})\s+", str((element.get("content") or {}).get("markdown") or ""))
        if match:
            authored_levels.add(len(match.group(1)))
    # The current enhanced API flattens detected headings to h1. If it returns
    # an authored hierarchy (or a non-h1 level), preserve that signal.
    if authored_levels != {1}:
        return {}

    proxies = sorted((_font_size_proxy(element) for _, element in headings), reverse=True)
    groups: list[float] = []
    for proxy in proxies:
        if proxy > 0 and (not groups or proxy < groups[-1] * _HEADING_SIZE_GROUP_RATIO):
            groups.append(proxy)

    levels: dict[int, int] = {}
    for index, element in headings:
        proxy = _font_size_proxy(element)
        if not groups or proxy <= 0:
            continue
        group_index = min(range(len(groups)), key=lambda candidate: abs(groups[candidate] - proxy))
        levels[index] = min(3, group_index + 1)
    return levels


def _promote_bold_title(markdown: str) -> str:
    """Separate short, title-cased leading bold text from its paragraph body."""
    bold_only = re.match(r"^\s*\*\*([^*\n]{2,100})\*\*\s*$", markdown, flags=re.DOTALL)
    if bold_only and len(bold_only.group(1).split()) <= 12:
        return f"### **{bold_only.group(1).strip()}**"

    leading = re.match(r"^\s*\*\*([^*\n]{2,100})\*\*\s+(.+)$", markdown, flags=re.DOTALL)
    if not leading:
        return markdown
    title = leading.group(1).strip()
    words = title.split()
    title_case_words = sum(1 for word in words if word[:1].isupper())
    if title.endswith((":", ";", ".", "?", "!")) or len(words) > 7:
        return markdown
    if title_case_words < max(1, math.ceil(len(words) * 0.6)):
        return markdown
    return f"### **{title}**\n\n{leading.group(2).strip()}"


def _code_language(markdown: str) -> str:
    body_match = re.search(r"```\s*\n([\s\S]*?)\n```", markdown)
    if not body_match:
        return ""
    body = body_match.group(1).strip()
    try:
        value = json.loads(body)
        if isinstance(value, (dict, list)):
            return "json"
    except (TypeError, ValueError):
        pass
    # Log exports sometimes prefix an otherwise JSON-shaped payload with a
    # timestamp and host address, so a full json.loads() is intentionally not
    # the only signal.
    if len(re.findall(r'(?m)^\s*"[^"\n]+"\s*:', body)) >= 2:
        return "json"
    if re.search(r"(?im)^\s*(?:program|subroutine|function|implicit\s+none)\b|\bend\s+program\b", body):
        return "fortran"
    cpp_signature = r"\b\w+(?:\.|->)\w+\([^)]*\b(?:int|float|double|char|bool)\s+\w+"
    if re.search(r"(?m)^\s*#include\s*[<\"]|\bstd::|\bint\s+main\s*\(", body) or re.search(cpp_signature, body):
        return "cpp"
    python_pattern = (
        r"(?m)(?:^\s*(?:from\s+\w+(?:\.\w+)*\s+import|import\s+\w+|"
        r"def\s+\w+\s*\(|class\s+\w+\s*[:(])|\bself\.|\bprint\s*\(|\bIn \[.*?\]:)"
    )
    if re.search(python_pattern, body):
        return "python"
    if re.search(r"(?m)^\s*(?:const|let|var)\s+\w+|=>|console\.log\s*\(", body):
        return "javascript"
    if re.search(r"(?im)^\s*(?:select|insert|update|delete|create)\b", body):
        return "sql"
    return ""


def normalize_elements(elements: list[dict[str, Any]]) -> list[str]:
    """Render one page of API elements for ParseBench evaluation."""
    heading_levels = _heading_levels(elements)
    rendered: list[str] = []
    for index, element in enumerate(elements):
        content = element.get("content") or {}
        category = str(element.get("category") or "").lower()
        if category == "table":
            markup = _semantic_table_headers(element_html(element))
        elif category == "chart":
            markup = _normalize_chart(element_html(element), _nearby_chart_title(elements, index))
        else:
            markup = str(content.get("markdown") or content.get("html") or html.escape(element_text(element)))
            if category.startswith("heading") and index in heading_levels:
                markup = re.sub(r"^\s*#{1,6}\s*", "#" * heading_levels[index] + " ", markup, count=1)
            elif category == "paragraph":
                markup = _promote_bold_title(markup)
            elif category == "code" and "```\n" in markup:
                language = _code_language(markup)
                if language:
                    markup = markup.replace("```\n", f"```{language}\n", 1)
        rendered.append(markup)
    return rendered
