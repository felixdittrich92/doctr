# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from html import escape as _html_escape
from typing import TYPE_CHECKING, Any, ClassVar, cast
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element as ETElement
from xml.etree.ElementTree import SubElement

import numpy as np

import doctr
from doctr.utils.common_types import BoundingBox

if TYPE_CHECKING:  # pragma: no cover
    from doctr.io.elements import Block, KIEPage, Line, Page, Table

__all__ = [
    "AsciiDocExporter",
    "DocumentExportsMixin",
    "HTMLExporter",
    "KIEPageExportsMixin",
    "MarkdownExporter",
    "PageExportsMixin",
    "page_reading_order",
]


def _export_as(exporters: dict[str, Any], format: str, **kwargs: Any) -> Any:
    fmt = format.strip().lower()
    if fmt not in exporters:
        raise ValueError(f"unsupported export format '{format}', should be one of {sorted(exporters)}")
    return exporters[fmt](**kwargs)


_LIST_LABELS = {"list_item"}
# Characters / line markers that carry a structural meaning and are escaped to preserve the raw OCR text
_MD_SPECIAL_CHARS = "\\`*_[]|#<>"
_MD_LINE_MARKERS = "-+>#=`"
_ADOC_LINE_MARKERS = "=*.-/+"


def _covering_region_indices(geoms: list[Any], region_geoms: list[Any], min_coverage: float = 0.5) -> list[int]:
    """For each element geometry, the index of the layout region covering the largest share of its area.

    Uses the same area-coverage criterion as :func:`doctr.models.reading_order.assign_layout_labels`, and
    returns -1 when no region covers the element by at least `min_coverage`. The geometries are expected to
    be in the same (upright) frame.
    """
    from doctr.models.reading_order.base import _to_boxes

    if len(region_geoms) == 0 or len(geoms) == 0:
        return [-1] * len(geoms)
    boxes, regions = _to_boxes(geoms), _to_boxes(region_geoms)
    inter_w = np.minimum(boxes[:, None, 2], regions[None, :, 2]) - np.maximum(boxes[:, None, 0], regions[None, :, 0])
    inter_h = np.minimum(boxes[:, None, 3], regions[None, :, 3]) - np.maximum(boxes[:, None, 1], regions[None, :, 1])
    inter = np.clip(inter_w, 0, None) * np.clip(inter_h, 0, None)
    areas = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9, None)
    coverage = inter / areas[:, None]
    best = coverage.argmax(axis=1)
    return [int(reg) if coverage[i, reg] >= min_coverage else -1 for i, reg in enumerate(best)]


def page_reading_order(page: "Page", direction: str = "auto") -> tuple[list[Any], list[str | None], str]:
    """Linearize the content of a page (blocks & tables) in reading order.

    Args:
        page: the page to linearize
        direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'

    Returns:
        a tuple with the ordered items (blocks & tables), their layout label (None without layout) and the
        effective reading direction
    """
    from doctr.io.elements import Block, Table
    from doctr.models.reading_order import (
        ReadingOrderPredictor,
        assign_layout_labels,
        deskew_reading_geometries,
        normalize_layout_label,
        resolve_reading_segments,
    )

    texts = [word.value for block in page.blocks for line in block.lines for word in line.words]
    language = page.language.get("value") if isinstance(page.language, dict) else None
    direction = ReadingOrderPredictor(direction=direction).resolve_direction(texts, language=language)
    region_geoms = [region.geometry for region in page.layout]
    region_labels = [region.type for region in page.layout]

    lines = [line for block in page.blocks for line in block.lines]
    elements: list[Any] = [*lines, *page.tables]
    if len(elements) == 0:
        return [], [], direction
    # De-skew once so labeling, ordering and region grouping share the same upright frame; the page angle is
    # estimated from the word polygons, which carry the detection model's true orientation
    elt_geoms, region_geoms = deskew_reading_geometries(
        [elt.geometry for elt in elements],
        region_geoms,
        page_shape=page.dimensions,
        angle_geoms=[word.geometry for line in lines for word in line.words],
    )
    elt_labels: list[str | None] = [None] * len(elements)
    if len(region_geoms) > 0:
        elt_labels = assign_layout_labels(elt_geoms, region_geoms, region_labels)
    elt_labels = ["Table" if isinstance(elt, Table) else label for elt, label in zip(elements, elt_labels)]
    segments = resolve_reading_segments(elt_geoms, direction=direction, labels=elt_labels)

    items: list[Any] = []
    labels: list[str | None] = []
    # Region index covering each element, used to group the lines of a wrapped list item under a single bullet
    region_idx = _covering_region_indices(elt_geoms, region_geoms) if len(region_geoms) > 0 else [-1] * len(elements)
    open_list_region: int | None = None  # region of the list bullet currently being built (None outside a list)
    for segment in segments:
        first = elements[segment[0]]
        seg_label = elt_labels[segment[0]]
        if isinstance(first, Table):
            items.append(first)
            labels.append("Table")
            open_list_region = None
            continue
        if normalize_layout_label(seg_label) in _LIST_LABELS:
            # One bullet per list-item region: consecutive lines sharing the same region are one bullet, so a
            # list item wrapped over several visual lines renders as a single bullet point.
            for idx in segment:
                region = region_idx[idx]
                if open_list_region is not None and region == open_list_region and region != -1:
                    items[-1] = Block(lines=[*items[-1].lines, elements[idx]])
                else:
                    items.append(Block(lines=[elements[idx]]))
                    labels.append(seg_label)
                    open_list_region = region
        else:
            items.append(Block(lines=[elements[idx] for idx in segment]))
            labels.append(seg_label)
            open_list_region = None
    return items, labels, direction


def _line_render_direction(line: "Line", page_direction: str, auto: bool) -> str:
    """Resolve the direction used to order the words of a line.

    For vertical pages the words are always read top to bottom. For horizontal pages, when the page direction
    was inferred automatically, the base direction of each line is detected from its own text so that an
    embedded left-to-right run (e.g. a Latin quotation on an Arabic page) keeps its natural word order; when
    the direction is set explicitly, it is applied uniformly to every line.
    """
    if page_direction in ("ttb-rtl", "ttb-ltr") or not auto or len(line.words) <= 1:
        return page_direction
    from doctr.models.reading_order import detect_text_direction

    return detect_text_direction([word.render() for word in line.words])


class _PageTextExporter:
    """Shared logic of the reading-order-aware text exporters.

    Subclasses define the format specifics: heading prefixes (per normalized layout label), the bullet
    prefix, character escaping, line finalization (neutralizing markers a line must not start with) and the
    table rendering.
    """

    headings: ClassVar[dict[str, str]] = {}
    bullet: ClassVar[str] = "- "
    page_break: ClassVar[str] = "\n\n"

    def escape_text(self, text: str) -> str:
        """Escape the characters carrying a structural meaning in the target format"""
        return text

    def finalize_line(self, line: str) -> str:
        """Neutralize the block-level markers a line must not start with in the target format"""
        return line

    def render_table(self, table: "Table", escape: bool = True) -> str:
        """Render a recognized table in the target format"""
        raise NotImplementedError

    def class_header(self, class_name: str, escape: bool = True) -> str:
        """Render the header of a detection class in a KIE export"""
        raise NotImplementedError

    def _line_text(self, line: "Line", direction: str, escape: bool) -> str:
        """Render the text of a line, ordering the words according to the reading direction."""
        words = line.words
        if direction in ("ttb-rtl", "ttb-ltr"):
            words = sorted(words, key=lambda word: float(np.asarray(word.geometry, dtype=np.float64)[..., 1].mean()))
        elif direction == "rtl":
            words = sorted(words, key=lambda word: -float(np.asarray(word.geometry, dtype=np.float64)[..., 0].mean()))
        text = " ".join(word.render() for word in words)
        return self.escape_text(text) if escape else text

    def export_page(
        self, page: "Page", direction: str = "auto", escape: bool = True, include_furniture: bool = True
    ) -> str:
        """Export a page, with its content sorted in reading order.

        Args:
            page: the page to export
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters or markers carrying a structural meaning should be neutralized
            include_furniture: whether page headers, page footers and footnotes should be included

        Returns:
            the exported page as a string
        """
        from doctr.io.elements import Table
        from doctr.models.reading_order import layout_label_role, normalize_layout_label

        auto = direction == "auto"
        items, labels, direction = page_reading_order(page, direction)
        parts: list[str] = []
        list_group: list[str] = []

        def _flush_list() -> None:
            if list_group:
                parts.append("\n".join(list_group))
                list_group.clear()

        for item, label in zip(items, labels):
            if not include_furniture and layout_label_role(label) in ("header", "footer", "footnote"):
                continue
            if isinstance(item, Table):
                _flush_list()
                rendered = self.render_table(item, escape=escape)
                if rendered:
                    parts.append(rendered)
                continue
            item_lines = [
                self._line_text(line, _line_render_direction(line, direction, auto), escape) for line in item.lines
            ]
            item_lines = [line for line in item_lines if line.strip()]
            if len(item_lines) == 0:
                continue
            norm_label = normalize_layout_label(label)
            if norm_label in self.headings:
                _flush_list()
                parts.append(self.headings[norm_label] + " ".join(item_lines))
            elif norm_label in _LIST_LABELS:
                # A list item (possibly wrapped over several lines) renders as a single bullet
                text = " ".join(item_lines)
                list_group.append(self.bullet + (self.finalize_line(text) if escape else text))
            else:
                _flush_list()
                parts.append("\n".join(self.finalize_line(line) if escape else line for line in item_lines))
        _flush_list()
        return "\n\n".join(parts)

    def export_kie_page(self, page: "KIEPage", direction: str = "auto", escape: bool = True) -> str:
        """Export a KIE page, with the predictions of each class sorted in reading order.

        Args:
            page: the KIE page to export
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters or markers carrying a structural meaning should be neutralized

        Returns:
            the exported page as a string, with one section per detection class
        """
        from doctr.models.reading_order import ReadingOrderPredictor

        language = page.language.get("value") if isinstance(page.language, dict) else None
        predictor = ReadingOrderPredictor(direction=direction)
        parts: list[str] = []
        for class_name, predictions in page.predictions.items():
            if len(predictions) == 0:
                continue
            order = predictor(
                [prediction.geometry for prediction in predictions],
                texts=[prediction.value for prediction in predictions],
                language=language,
                page_shape=page.dimensions,
            )
            values = "\n".join(
                self.bullet
                + (self.finalize_line(self.escape_text(predictions[idx].value)) if escape else predictions[idx].value)
                for idx in order
            )
            parts.append(f"{self.class_header(class_name, escape)}\n\n{values}")
        return "\n\n".join(parts)

    def export_document(self, document: Any, page_break: str | None = None, **kwargs: Any) -> str:
        """Export a document page by page.

        Args:
            document: the document to export
            page_break: the string inserted between two pages (a format-specific default when None)
            **kwargs: additional keyword arguments passed to the page export

        Returns:
            the exported document as a string
        """
        from doctr.io.elements import KIEPage

        page_break = self.page_break if page_break is None else page_break
        return page_break.join(
            self.export_kie_page(page, **kwargs) if isinstance(page, KIEPage) else self.export_page(page, **kwargs)
            for page in document.pages
        )


class MarkdownExporter(_PageTextExporter):
    """Export OCR results to Markdown, with the content sorted in reading order.

    >>> from doctr.io import MarkdownExporter
    >>> markdown = MarkdownExporter().export_page(page)  # doctest: +SKIP
    """

    headings: ClassVar[dict[str, str]] = {"title": "# ", "section_header": "## "}
    bullet: ClassVar[str] = "- "
    page_break: ClassVar[str] = "\n\n---\n\n"

    def escape_text(self, text: str) -> str:
        return "".join(f"\\{char}" if char in _MD_SPECIAL_CHARS else char for char in text)

    def finalize_line(self, line: str) -> str:
        stripped = line.lstrip()
        if stripped and (stripped[0] in _MD_LINE_MARKERS or stripped.split(" ")[0].rstrip(".").isdigit()):
            return f"\\{line}" if line[0] != "\\" else line
        return line

    def render_table(self, table: "Table", escape: bool = True) -> str:
        """Render a table as a GitHub-flavored Markdown table (first row used as header)"""
        grid = table.to_grid()
        if len(grid) == 0 or len(grid[0]) == 0:
            return ""

        def _cell(value: str) -> str:
            value = self.escape_text(value) if escape else value.replace("|", "\\|")
            return value.replace("\n", " ").strip()

        rows = ["| " + " | ".join(_cell(value) for value in row) + " |" for row in grid]
        separator = "| " + " | ".join("---" for _ in grid[0]) + " |"
        return "\n".join([rows[0], separator, *rows[1:]])

    def class_header(self, class_name: str, escape: bool = True) -> str:
        return f"**{self.escape_text(class_name) if escape else class_name}**"


class AsciiDocExporter(_PageTextExporter):
    """Export OCR results to AsciiDoc, with the content sorted in reading order.

    >>> from doctr.io import AsciiDocExporter
    >>> asciidoc = AsciiDocExporter().export_page(page)  # doctest: +SKIP
    """

    headings: ClassVar[dict[str, str]] = {"title": "== ", "section_header": "=== "}
    bullet: ClassVar[str] = "* "
    page_break: ClassVar[str] = "\n\n<<<\n\n"

    def finalize_line(self, line: str) -> str:
        stripped = line.lstrip()
        if stripped and stripped[0] in _ADOC_LINE_MARKERS:
            return f"{{empty}}{line}"
        return line

    def render_table(self, table: "Table", escape: bool = True) -> str:
        """Render a table as an AsciiDoc table (first row used as header)"""
        grid = table.to_grid()
        if len(grid) == 0 or len(grid[0]) == 0:
            return ""

        def _row(row: list[str]) -> str:
            return " ".join("|" + value.replace("|", "\\|").replace("\n", " ").strip() for value in row)

        return "\n".join(["|===", _row(grid[0]), "", *[_row(row) for row in grid[1:]], "|==="])

    def class_header(self, class_name: str, escape: bool = True) -> str:
        return f"*{class_name}*"


class HTMLExporter(_PageTextExporter):
    """Export OCR results to semantic HTML, with the content sorted in reading order.

    Headings map to `<h1>`/`<h2>`, list items to `<ul><li>`, recognized tables to `<table>` and
    paragraphs to `<p>` (with `<br>` between the visual lines of a paragraph).

    >>> from doctr.io import HTMLExporter
    >>> html = HTMLExporter().export_page(page)  # doctest: +SKIP
    """

    headings: ClassVar[dict[str, str]] = {"title": "h1", "section_header": "h2"}
    page_break: ClassVar[str] = "\n<hr>\n"

    def escape_text(self, text: str) -> str:
        return _html_escape(text, quote=False)

    def export_page(
        self, page: "Page", direction: str = "auto", escape: bool = True, include_furniture: bool = True
    ) -> str:
        from doctr.io.elements import Table
        from doctr.models.reading_order import layout_label_role, normalize_layout_label

        auto = direction == "auto"
        items, labels, direction = page_reading_order(page, direction)
        parts: list[str] = []
        list_group: list[str] = []

        def _flush_list() -> None:
            if list_group:
                parts.append("<ul>\n" + "\n".join(list_group) + "\n</ul>")
                list_group.clear()

        for item, label in zip(items, labels):
            if not include_furniture and layout_label_role(label) in ("header", "footer", "footnote"):
                continue
            if isinstance(item, Table):
                _flush_list()
                rendered = self.render_table(item, escape=escape)
                if rendered:
                    parts.append(rendered)
                continue
            item_lines = [
                self._line_text(line, _line_render_direction(line, direction, auto), escape) for line in item.lines
            ]
            item_lines = [line for line in item_lines if line.strip()]
            if len(item_lines) == 0:
                continue
            norm_label = normalize_layout_label(label)
            if norm_label in self.headings:
                _flush_list()
                tag = self.headings[norm_label]
                parts.append(f"<{tag}>{' '.join(item_lines)}</{tag}>")
            elif norm_label in _LIST_LABELS:
                list_group.append(f"<li>{' '.join(item_lines)}</li>")
            else:
                _flush_list()
                parts.append("<p>" + "<br>\n".join(item_lines) + "</p>")
        _flush_list()
        return "\n".join(parts)

    def render_table(self, table: "Table", escape: bool = True) -> str:
        """Render a table as an HTML table (first row used as header)"""
        grid = table.to_grid()
        if len(grid) == 0 or len(grid[0]) == 0:
            return ""

        def _cell(value: str, tag: str) -> str:
            content = self.escape_text(value) if escape else value
            return f"<{tag}>{content.strip()}</{tag}>"

        head = "<tr>" + "".join(_cell(value, "th") for value in grid[0]) + "</tr>"
        body = "\n".join("<tr>" + "".join(_cell(value, "td") for value in row) + "</tr>" for row in grid[1:])
        return f"<table>\n{head}\n{body}\n</table>" if body else f"<table>\n{head}\n</table>"

    def export_kie_page(self, page: "KIEPage", direction: str = "auto", escape: bool = True) -> str:
        from doctr.models.reading_order import ReadingOrderPredictor

        language = page.language.get("value") if isinstance(page.language, dict) else None
        predictor = ReadingOrderPredictor(direction=direction)
        parts: list[str] = []
        for class_name, predictions in page.predictions.items():
            if len(predictions) == 0:
                continue
            order = predictor(
                [prediction.geometry for prediction in predictions],
                texts=[prediction.value for prediction in predictions],
                language=language,
                page_shape=page.dimensions,
            )
            values = "\n".join(
                f"<li>{self.escape_text(predictions[idx].value) if escape else predictions[idx].value}</li>"
                for idx in order
            )
            header = self.escape_text(class_name) if escape else class_name
            parts.append(f"<h3>{header}</h3>\n<ul>\n{values}\n</ul>")
        return "\n".join(parts)


def _resolve_hocr_language(language: dict[str, Any]) -> str:
    """Resolve the language code to use in the hOCR export, falling back to 'en'.

    Args:
        language: the page language dictionary `{"value": str | None, "confidence": float | None}`

    Returns:
        the detected language code when available, 'en' otherwise
    """
    lang_value = language.get("value") if isinstance(language, dict) else None
    return lang_value if isinstance(lang_value, str) and len(lang_value) > 0 else "en"


def _hocr_bbox(geometry: BoundingBox, width: int, height: int) -> str:
    """Format a relative straight bounding box as an absolute hOCR `bbox` property string.

    Args:
        geometry: the relative bounding box ((xmin, ymin), (xmax, ymax))
        width: the page width in pixels
        height: the page height in pixels

    Returns:
        the hOCR `bbox` property string
    """
    (xmin, ymin), (xmax, ymax) = geometry
    return (
        f"bbox {int(round(xmin * width))} {int(round(ymin * height))} "
        f"{int(round(xmax * width))} {int(round(ymax * height))}"
    )


class PageExportsMixin:
    """Export functionality of a :class:`~doctr.io.elements.Page`"""

    if TYPE_CHECKING:  # structural attributes provided by the element class
        page: np.ndarray
        blocks: list["Block"]
        page_idx: int
        dimensions: tuple[int, int]
        orientation: dict[str, Any]
        language: dict[str, Any]
        layout: list[Any]
        tables: list["Table"]

        def export(self) -> dict[str, Any]: ...

    def render(self, block_break: str = "\n\n") -> str:
        """Renders the full text of the element"""
        return block_break.join(b.render() for b in self.blocks)

    def export_as_xml(self, file_title: str = "docTR - XML export (hOCR)") -> tuple[bytes, ET.ElementTree]:
        """Export the page as XML (hOCR-format)
        convention: https://github.com/kba/hocr-spec/blob/master/1.2/spec.md

        Args:
            file_title: the title of the XML file

        Returns:
            a tuple of the XML byte string, and its ElementTree
        """
        p_idx = self.page_idx
        block_count: int = 1
        line_count: int = 1
        word_count: int = 1
        height, width = self.dimensions
        language = _resolve_hocr_language(self.language)
        # Create the XML root element
        page_hocr = ETElement("html", attrib={"xmlns": "http://www.w3.org/1999/xhtml", "xml:lang": str(language)})
        # Create the header / SubElements of the root element
        head = SubElement(page_hocr, "head")
        SubElement(head, "title").text = file_title
        SubElement(head, "meta", attrib={"http-equiv": "Content-Type", "content": "text/html; charset=utf-8"})
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-system", "content": f"python-doctr {doctr.__version__}"},  # type: ignore[attr-defined]
        )
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-capabilities", "content": "ocr_page ocr_carea ocr_par ocr_line ocrx_word"},
        )
        # Create the body
        body = SubElement(page_hocr, "body")
        page_div = SubElement(
            body,
            "div",
            attrib={
                "class": "ocr_page",
                "id": f"page_{p_idx + 1}",
                "title": f"image; bbox 0 0 {width} {height}; ppageno 0",
            },
        )
        # iterate over the blocks / lines / words and create the XML elements in body line by line with the attributes
        for block in self.blocks:
            if len(block.geometry) != 2:
                raise TypeError("XML export is only available for straight bounding boxes for now.")
            block_bbox = _hocr_bbox(block.geometry, width, height)  # type: ignore[arg-type]
            block_div = SubElement(
                page_div,
                "div",
                attrib={
                    "class": "ocr_carea",
                    "id": f"block_{block_count}",
                    "title": block_bbox,
                },
            )
            paragraph = SubElement(
                block_div,
                "p",
                attrib={
                    "class": "ocr_par",
                    "id": f"par_{block_count}",
                    "title": block_bbox,
                },
            )
            block_count += 1
            for line in block.lines:
                # NOTE: baseline, x_size, x_descenders, x_ascenders is currently initalized to 0
                line_span = SubElement(
                    paragraph,
                    "span",
                    attrib={
                        "class": "ocr_line",
                        "id": f"line_{line_count}",
                        "title": (
                            f"{_hocr_bbox(line.geometry, width, height)}; "  # type: ignore[arg-type]
                            "baseline 0 0; x_size 0; x_descenders 0; x_ascenders 0"
                        ),
                    },
                )
                line_count += 1
                for word in line.words:
                    conf = word.confidence
                    word_div = SubElement(
                        line_span,
                        "span",
                        attrib={
                            "class": "ocrx_word",
                            "id": f"word_{word_count}",
                            "title": (
                                f"{_hocr_bbox(word.geometry, width, height)}; "  # type: ignore[arg-type]
                                f"x_wconf {int(round(conf * 100))}"
                            ),
                        },
                    )
                    # set the text
                    word_div.text = word.value
                    word_count += 1

        return (ET.tostring(page_hocr, encoding="utf-8", method="xml"), ET.ElementTree(page_hocr))

    def items_in_reading_order(self, direction: str = "auto") -> list["Block | Table"]:
        """Return the content of the page (blocks & tables) sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'

        Returns:
            list of blocks & tables in reading order
        """
        return page_reading_order(cast("Page", self), direction)[0]

    def export_as_markdown(self, direction: str = "auto", escape: bool = True, include_furniture: bool = True) -> str:
        """Export the page as Markdown, with its content sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters carrying a structural meaning in Markdown should be escaped
            include_furniture: whether page headers, page footers and footnotes should be included

        Returns:
            a Markdown string
        """
        return MarkdownExporter().export_page(
            cast("Page", self), direction=direction, escape=escape, include_furniture=include_furniture
        )

    def export_as_asciidoc(self, direction: str = "auto", escape: bool = True, include_furniture: bool = True) -> str:
        """Export the page as AsciiDoc, with its content sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the line-level markers carrying a structural meaning in AsciiDoc should be neutralized
            include_furniture: whether page headers, page footers and footnotes should be included

        Returns:
            an AsciiDoc string
        """
        return AsciiDocExporter().export_page(
            cast("Page", self), direction=direction, escape=escape, include_furniture=include_furniture
        )

    def export_as_html(self, direction: str = "auto", include_furniture: bool = True) -> str:
        """Export the page as semantic HTML, with its content sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            include_furniture: whether page headers, page footers and footnotes should be included

        Returns:
            an HTML string
        """
        return HTMLExporter().export_page(cast("Page", self), direction=direction, include_furniture=include_furniture)

    def export_as(self, format: str, **kwargs: Any) -> Any:
        """Export the page in the requested format.

        Args:
            format: one of 'markdown'/'md', 'asciidoc'/'adoc', 'text'/'txt', 'json'/'dict', 'xml'/'hocr'
            **kwargs: additional keyword arguments passed to the format-specific export method

        Returns:
            the exported page
        """
        exporters: dict[str, Any] = {
            "markdown": self.export_as_markdown,
            "md": self.export_as_markdown,
            "asciidoc": self.export_as_asciidoc,
            "adoc": self.export_as_asciidoc,
            "html": self.export_as_html,
            "text": self.render,
            "txt": self.render,
            "json": lambda: self.export(),
            "dict": lambda: self.export(),
            "xml": self.export_as_xml,
            "hocr": self.export_as_xml,
        }
        return _export_as(exporters, format, **kwargs)


class KIEPageExportsMixin:
    """Export functionality of a :class:`~doctr.io.elements.KIEPage`"""

    if TYPE_CHECKING:  # structural attributes provided by the element class
        page: np.ndarray
        predictions: dict[str, list[Any]]
        page_idx: int
        dimensions: tuple[int, int]
        orientation: dict[str, Any]
        language: dict[str, Any]

        def export(self) -> dict[str, Any]: ...

    def render(self, prediction_break: str = "\n\n") -> str:
        """Renders the full text of the element"""
        return prediction_break.join(
            f"{class_name}: {p.render()}" for class_name, predictions in self.predictions.items() for p in predictions
        )

    def export_as_xml(self, file_title: str = "docTR - XML export (hOCR)") -> tuple[bytes, ET.ElementTree]:
        """Export the page as XML (hOCR-format)
        convention: https://github.com/kba/hocr-spec/blob/master/1.2/spec.md

        Args:
            file_title: the title of the XML file

        Returns:
            a tuple of the XML byte string, and its ElementTree
        """
        p_idx = self.page_idx
        prediction_count: int = 1
        height, width = self.dimensions
        language = _resolve_hocr_language(self.language)
        # Create the XML root element
        page_hocr = ETElement("html", attrib={"xmlns": "http://www.w3.org/1999/xhtml", "xml:lang": str(language)})
        # Create the header / SubElements of the root element
        head = SubElement(page_hocr, "head")
        SubElement(head, "title").text = file_title
        SubElement(head, "meta", attrib={"http-equiv": "Content-Type", "content": "text/html; charset=utf-8"})
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-system", "content": f"python-doctr {doctr.__version__}"},  # type: ignore[attr-defined]
        )
        SubElement(
            head,
            "meta",
            attrib={"name": "ocr-capabilities", "content": "ocr_page ocr_carea ocr_par ocr_line ocrx_word"},
        )
        # Create the body
        body = SubElement(page_hocr, "body")
        SubElement(
            body,
            "div",
            attrib={
                "class": "ocr_page",
                "id": f"page_{p_idx + 1}",
                "title": f"image; bbox 0 0 {width} {height}; ppageno 0",
            },
        )
        # iterate over the blocks / lines / words and create the XML elements in body line by line with the attributes
        for class_name, predictions in self.predictions.items():
            for prediction in predictions:
                if len(prediction.geometry) != 2:
                    raise TypeError("XML export is only available for straight bounding boxes for now.")
                prediction_bbox = _hocr_bbox(prediction.geometry, width, height)
                prediction_div = SubElement(
                    body,
                    "div",
                    attrib={
                        "class": "ocr_carea",
                        "id": f"{class_name}_prediction_{prediction_count}",
                        "title": prediction_bbox,
                    },
                )
                # NOTE: ocr_par, ocr_line and ocrx_word are the same because the KIE predictions contain only words
                # This is a workaround to make it PDF/A compatible
                par_div = SubElement(
                    prediction_div,
                    "p",
                    attrib={
                        "class": "ocr_par",
                        "id": f"{class_name}_par_{prediction_count}",
                        "title": prediction_bbox,
                    },
                )
                line_span = SubElement(
                    par_div,
                    "span",
                    attrib={
                        "class": "ocr_line",
                        "id": f"{class_name}_line_{prediction_count}",
                        "title": f"{prediction_bbox}; baseline 0 0; x_size 0; x_descenders 0; x_ascenders 0",
                    },
                )
                word_div = SubElement(
                    line_span,
                    "span",
                    attrib={
                        "class": "ocrx_word",
                        "id": f"{class_name}_word_{prediction_count}",
                        "title": f"{prediction_bbox}; x_wconf {int(round(prediction.confidence * 100))}",
                    },
                )
                word_div.text = prediction.value
                prediction_count += 1

        return ET.tostring(page_hocr, encoding="utf-8", method="xml"), ET.ElementTree(page_hocr)

    def export_as_markdown(self, direction: str = "auto", escape: bool = True) -> str:
        """Export the KIE page as Markdown, with the predictions of each class sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the characters carrying a structural meaning in Markdown should be escaped

        Returns:
            a Markdown string with one section per detection class
        """
        return MarkdownExporter().export_kie_page(cast("KIEPage", self), direction=direction, escape=escape)

    def export_as_asciidoc(self, direction: str = "auto", escape: bool = True) -> str:
        """Export the KIE page as AsciiDoc, with the predictions of each class sorted in reading order.

        Args:
            direction: reading direction, one of 'auto', 'ltr', 'rtl', 'ttb-rtl' or 'ttb-ltr'
            escape: whether the line-level markers carrying a structural meaning in AsciiDoc should be neutralized

        Returns:
            an AsciiDoc string with one section per detection class
        """
        return AsciiDocExporter().export_kie_page(cast("KIEPage", self), direction=direction, escape=escape)

    def export_as_html(self, direction: str = "auto") -> str:
        """Export the KIE page as semantic HTML, with the predictions of each class sorted in reading order"""
        return HTMLExporter().export_kie_page(cast("KIEPage", self), direction=direction)

    def export_as(self, format: str, **kwargs: Any) -> Any:
        """Export the KIE page in the requested format ('markdown'/'md', 'asciidoc'/'adoc', 'text'/'txt',
        'json'/'dict', 'xml'/'hocr')."""
        exporters: dict[str, Any] = {
            "markdown": self.export_as_markdown,
            "md": self.export_as_markdown,
            "asciidoc": self.export_as_asciidoc,
            "adoc": self.export_as_asciidoc,
            "html": self.export_as_html,
            "text": self.render,
            "txt": self.render,
            "json": lambda: self.export(),
            "dict": lambda: self.export(),
            "xml": self.export_as_xml,
            "hocr": self.export_as_xml,
        }
        return _export_as(exporters, format, **kwargs)


class DocumentExportsMixin:
    """Export functionality of a :class:`~doctr.io.elements.Document` (also used by `KIEDocument`)"""

    if TYPE_CHECKING:  # structural attributes provided by the element class
        pages: list[Any]

        def export(self) -> dict[str, Any]: ...

    def render(self, page_break: str = "\n\n\n\n") -> str:
        """Renders the full text of the element"""
        return page_break.join(p.render() for p in self.pages)

    def export_as_xml(self, **kwargs) -> list[tuple[bytes, ET.ElementTree]]:
        """Export the document as XML (hOCR-format)

        Args:
            **kwargs: additional keyword arguments passed to the Page.export_as_xml method

        Returns:
            list of tuple of (bytes, ElementTree)
        """
        return [page.export_as_xml(**kwargs) for page in self.pages]

    def export_as_markdown(self, page_break: str = "\n\n---\n\n", **kwargs: Any) -> str:
        """Export the document as Markdown, with the content of each page sorted in reading order.

        Args:
            page_break: the string inserted between two pages (a thematic break by default)
            **kwargs: additional keyword arguments passed to the `Page.export_as_markdown` method

        Returns:
            a Markdown string
        """
        return page_break.join(page.export_as_markdown(**kwargs) for page in self.pages)

    def export_as_asciidoc(self, page_break: str = "\n\n<<<\n\n", **kwargs: Any) -> str:
        """Export the document as AsciiDoc, with the content of each page sorted in reading order.

        Args:
            page_break: the string inserted between two pages (an AsciiDoc page break by default)
            **kwargs: additional keyword arguments passed to the `Page.export_as_asciidoc` method

        Returns:
            an AsciiDoc string
        """
        return page_break.join(page.export_as_asciidoc(**kwargs) for page in self.pages)

    def export_as_html(self, page_break: str = "<hr>", **kwargs: Any) -> str:
        """Export the document as semantic HTML, with the content of each page sorted in reading order.

        Args:
            page_break: the HTML snippet inserted between two pages
            **kwargs: additional keyword arguments passed to the page export

        Returns:
            an HTML string
        """
        return page_break.join(page.export_as_html(**kwargs) for page in self.pages)

    def export_as(self, format: str, **kwargs: Any) -> Any:
        """Export the document in the requested format ('markdown'/'md', 'asciidoc'/'adoc', 'text'/'txt',
        'json'/'dict', 'xml'/'hocr')."""
        exporters: dict[str, Any] = {
            "markdown": self.export_as_markdown,
            "md": self.export_as_markdown,
            "asciidoc": self.export_as_asciidoc,
            "adoc": self.export_as_asciidoc,
            "html": self.export_as_html,
            "text": self.render,
            "txt": self.render,
            "json": lambda: self.export(),
            "dict": lambda: self.export(),
            "xml": self.export_as_xml,
            "hocr": self.export_as_xml,
        }
        return _export_as(exporters, format, **kwargs)
