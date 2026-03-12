from __future__ import annotations

import html
import math
import os
import posixpath
import re
from datetime import datetime

BEGIN_TOC = "{{ BEGIN_TOC }}"
END_TOC = "{{ END_TOC }}"

TAG_INFO = {
    "note": ("课程笔记", "note"),
    "review": ("复习资料", "review"),
    "lab": ("实验报告", "lab"),
    "exam": ("历年试题", "exam"),
}


def on_page_markdown(markdown, page, config, files):
    if BEGIN_TOC not in markdown or END_TOC not in markdown:
        return markdown

    block_pattern = re.compile(
        r"\{\{\s*BEGIN_TOC\s*\}\}(.*?)\{\{\s*END_TOC\s*\}\}",
        flags=re.DOTALL,
    )

    def replace_block(match):
        toc_block = match.group(1)
        rendered = _render_toc_block(toc_block, page, config, files)
        return rendered if rendered else match.group(0)

    return block_pattern.sub(replace_block, markdown)


def _render_toc_block(toc_block, page, config, files):
    groups = []
    current = None

    for raw_line in toc_block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        item_match = re.match(r"^(?P<indent>\s*)-\s+(?P<body>.+)$", line)
        if not item_match:
            continue

        indent = len(item_match.group("indent").replace("\t", "    "))
        body = item_match.group("body").strip()

        if indent == 0:
            title = body[:-1].strip() if body.endswith(":") else body
            current = {"title": title, "items": []}
            groups.append(current)
            continue

        if current is None:
            continue

        entry = _parse_entry(body)
        if entry:
            current["items"].append(entry)

    groups = [g for g in groups if g["items"]]
    if not groups:
        return ""

    html_parts = ['<div class="toc-board">']

    for group in groups:
        html_parts.append('<details class="toc-group" open>')
        html_parts.append(
            '<summary><span class="toc-group__title">{}</span></summary>'.format(
                html.escape(group["title"])
            )
        )
        html_parts.append('<div class="toc-group__items">')

        for item in group["items"]:
            source_file = _resolve_source_file(item["href"], page, files)
            stats = _collect_file_stats(source_file, config)
            meta_html = _build_meta(stats)

            chips_html = _build_chips(item["tags"])
            updated = _format_relative_time(stats["modified"]) if stats else ""

            html_parts.append(
                (
                    '<a class="toc-entry" href="{href}">'
                    '<span class="toc-entry__main">'
                    '<span class="toc-entry__title">{title}</span>'
                    '<span class="toc-entry__meta">{meta}</span>'
                    "</span>"
                    '<span class="toc-entry__tags">{chips}</span>'
                    '<span class="toc-entry__time">{updated}</span>'
                    "</a>"
                ).format(
                    href=html.escape(item["href"], quote=True),
                    title=html.escape(item["title"]),
                    meta=meta_html,
                    chips=chips_html,
                    updated=html.escape(updated),
                )
            )

        html_parts.append("</div>")
        html_parts.append("</details>")

    html_parts.append("</div>")
    return "\n".join(html_parts)


def _parse_entry(body):
    match = re.match(
        r"^(?P<title>.+?)\s*(?:\[(?P<tags>[^\]]+)\])?\s*:\s*(?P<href>\S+)\s*$",
        body,
    )
    if not match:
        return None

    raw_tags = match.group("tags") or ""
    tags = []
    for tag in raw_tags.split(","):
        name = tag.strip().lower()
        if name:
            tags.append(name)

    return {
        "title": match.group("title").strip(),
        "href": match.group("href").strip(),
        "tags": tags,
    }


def _build_chips(tags):
    if not tags:
        return ""

    chips = []
    for tag in tags:
        label, css_name = TAG_INFO.get(tag, (tag, "default"))
        chips.append(
            '<span class="toc-pill toc-pill--{css}">{label}</span>'.format(
                css=html.escape(css_name),
                label=html.escape(label),
            )
        )
    return "".join(chips)


def _build_meta(stats):
    if not stats:
        return '<span class="toc-meta-item toc-meta-item--empty">暂无统计</span>'

    items = [
        ("◔", str(stats["words"]), "toc-meta-item--words"),
    ]
    if stats["code_lines"] > 0:
        items.append(("</>", str(stats["code_lines"]), "toc-meta-item--code"))
    items.append(("◷", "{} mins".format(stats["read_minutes"]), "toc-meta-item--time"))

    parts = []
    for icon, value, css_class in items:
        parts.append(
            (
                '<span class="toc-meta-item {css}">'
                '<span class="toc-meta-icon">{icon}</span>'
                '<span class="toc-meta-value">{value}</span>'
                "</span>"
            ).format(
                css=css_class,
                icon=html.escape(icon),
                value=html.escape(value),
            )
        )
    return "".join(parts)


def _resolve_source_file(href, page, files):
    href = href.split("#", 1)[0].split("?", 1)[0].strip()
    if not href:
        return None
    if re.match(r"^(https?:|mailto:|#)", href, flags=re.IGNORECASE):
        return None

    base_dir = posixpath.dirname(page.file.src_uri)
    normalized = posixpath.normpath(posixpath.join(base_dir, href.lstrip("/")))

    candidates = []
    if posixpath.splitext(normalized)[1]:
        candidates.append(normalized)
    else:
        trimmed = normalized.rstrip("/")
        candidates.append(trimmed + ".md")
        candidates.append(posixpath.join(trimmed, "index.md"))

    for candidate in candidates:
        file_obj = files.get_file_from_path(candidate)
        if file_obj:
            return candidate
    return None


def _collect_file_stats(src_uri, config):
    if not src_uri:
        return None

    docs_dir = config.get("docs_dir")
    if not docs_dir:
        return None

    abs_path = os.path.join(docs_dir, *src_uri.split("/"))
    if not os.path.exists(abs_path):
        return None

    try:
        with open(abs_path, "r", encoding="utf-8") as file_handle:
            content = file_handle.read()
    except OSError:
        return None

    words = _count_words(content)
    code_lines = _count_code_lines(content)
    read_minutes = max(1, math.ceil(words / 256 + code_lines / 80))
    modified = os.path.getmtime(abs_path)

    return {
        "words": words,
        "code_lines": code_lines,
        "read_minutes": read_minutes,
        "modified": modified,
    }


def _count_words(text):
    text = re.sub(r"^---\s*.*?\s*---\s*", "", text, flags=re.DOTALL)
    no_code_lines = []
    in_fence = False

    for line in text.splitlines():
        striped = line.strip()
        if striped.startswith("```") or striped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if not in_fence:
            no_code_lines.append(line)

    plain_text = "\n".join(no_code_lines)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", plain_text))
    en_count = len(re.findall(r"[A-Za-z0-9_]+", plain_text))
    return cjk_count + en_count


def _count_code_lines(text):
    count = 0
    in_fence = False

    for line in text.splitlines():
        striped = line.strip()
        if striped.startswith("```") or striped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence and striped:
            count += 1
    return count


def _format_relative_time(timestamp):
    now = datetime.now().timestamp()
    delta_days = max(0, int((now - timestamp) / 86400))

    if delta_days <= 0:
        return "today"
    if delta_days < 30:
        return "{} days ago".format(delta_days)

    months = delta_days // 30
    if months < 12:
        return "{} months ago".format(months)

    years = delta_days // 365
    return "{} years ago".format(years)
