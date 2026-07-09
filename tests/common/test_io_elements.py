from xml.etree.ElementTree import ElementTree

import numpy as np
import pandas as pd
import pytest

from doctr.file_utils import CLASS_NAME
from doctr.io import elements


def _mock_words(size=(1.0, 1.0), offset=(0, 0), confidence=0.9, objectness_score=0.9):
    return [
        elements.Word(
            "hello",
            confidence,
            ((offset[0], offset[1]), (size[0] / 2 + offset[0], size[1] / 2 + offset[1])),
            objectness_score,
            {"value": 0, "confidence": None},
        ),
        elements.Word(
            "world",
            confidence,
            ((size[0] / 2 + offset[0], size[1] / 2 + offset[1]), (size[0] + offset[0], size[1] + offset[1])),
            objectness_score,
            {"value": 0, "confidence": None},
        ),
    ]


def _mock_artefacts(size=(1, 1), offset=(0, 0), confidence=0.8):
    sub_size = (size[0] / 2, size[1] / 2)
    return [
        elements.Artefact(
            "qr_code", confidence, ((offset[0], offset[1]), (sub_size[0] + offset[0], sub_size[1] + offset[1]))
        ),
        elements.Artefact(
            "qr_code",
            confidence,
            ((sub_size[0] + offset[0], sub_size[1] + offset[1]), (size[0] + offset[0], size[1] + offset[1])),
        ),
    ]


def _mock_layout():
    return [
        elements.LayoutElement("Title", 0.95, ((0.1, 0.05), (0.9, 0.15))),
        elements.LayoutElement("Text", 0.88, ((0.1, 0.2), (0.9, 0.9))),
    ]


def _mock_lines(size=(1, 1), offset=(0, 0)):
    sub_size = (size[0] / 2, size[1] / 2)
    return [
        elements.Line(_mock_words(size=sub_size, offset=offset)),
        elements.Line(_mock_words(size=sub_size, offset=(offset[0] + sub_size[0], offset[1] + sub_size[1]))),
    ]


def _mock_prediction(size=(1.0, 1.0), offset=(0, 0), confidence=0.9, objectness_score=0.9):
    return [
        elements.Prediction(
            "hello",
            confidence,
            ((offset[0], offset[1]), (size[0] / 2 + offset[0], size[1] / 2 + offset[1])),
            objectness_score,
            {"value": 0, "confidence": None},
        ),
        elements.Prediction(
            "world",
            confidence,
            ((size[0] / 2 + offset[0], size[1] / 2 + offset[1]), (size[0] + offset[0], size[1] + offset[1])),
            objectness_score,
            {"value": 0, "confidence": None},
        ),
    ]


def _mock_blocks(size=(1, 1), offset=(0, 0)):
    sub_size = (size[0] / 4, size[1] / 4)
    return [
        elements.Block(
            _mock_lines(size=sub_size, offset=offset),
            _mock_artefacts(size=sub_size, offset=(offset[0] + sub_size[0], offset[1] + sub_size[1])),
        ),
        elements.Block(
            _mock_lines(size=sub_size, offset=(offset[0] + 2 * sub_size[0], offset[1] + 2 * sub_size[1])),
            _mock_artefacts(size=sub_size, offset=(offset[0] + 3 * sub_size[0], offset[1] + 3 * sub_size[1])),
        ),
    ]


def _mock_pages(block_size=(1, 1), block_offset=(0, 0)):
    return [
        elements.Page(
            np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8),
            _mock_blocks(block_size, block_offset),
            0,
            (300, 200),
            {"value": 0.0, "confidence": 1.0},
            {"value": "EN", "confidence": 0.8},
        ),
        elements.Page(
            np.random.randint(0, 255, (500, 1000, 3), dtype=np.uint8),
            _mock_blocks(block_size, block_offset),
            1,
            (500, 1000),
            {"value": 0.15, "confidence": 0.8},
            {"value": "FR", "confidence": 0.7},
        ),
    ]


def _mock_kie_pages(prediction_size=(1, 1), prediction_offset=(0, 0)):
    return [
        elements.KIEPage(
            np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8),
            {CLASS_NAME: _mock_prediction(prediction_size, prediction_offset)},
            0,
            (300, 200),
            {"value": 0.0, "confidence": 1.0},
            {"value": "EN", "confidence": 0.8},
        ),
        elements.KIEPage(
            np.random.randint(0, 255, (500, 1000, 3), dtype=np.uint8),
            {CLASS_NAME: _mock_prediction(prediction_size, prediction_offset)},
            1,
            (500, 1000),
            {"value": 0.15, "confidence": 0.8},
            {"value": "FR", "confidence": 0.7},
        ),
    ]


def test_element():
    with pytest.raises(KeyError):
        elements.Element(sub_elements=[1])


def test_word():
    word_str = "hello"
    conf = 0.8
    objectness_score = 0.9
    geom = ((0, 0), (1, 1))
    crop_orientation = {"value": 0, "confidence": None}
    word = elements.Word(word_str, conf, geom, objectness_score, crop_orientation)

    # Attribute checks
    assert word.value == word_str
    assert word.confidence == conf
    assert word.geometry == geom
    assert word.objectness_score == objectness_score
    assert word.crop_orientation == crop_orientation

    # Render
    assert word.render() == word_str

    # Export
    assert word.export() == {
        "value": word_str,
        "confidence": conf,
        "geometry": geom,
        "objectness_score": objectness_score,
        "crop_orientation": crop_orientation,
    }

    # Repr
    assert word.__repr__() == f"Word(value='hello', confidence={conf:.2})"

    # Class method
    state_dict = {
        "value": "there",
        "confidence": 0.1,
        "geometry": ((0, 0), (0.5, 0.5)),
        "objectness_score": objectness_score,
        "crop_orientation": crop_orientation,
    }
    word = elements.Word.from_dict(state_dict)
    assert word.export() == state_dict


def test_line():
    geom = ((0, 0), (0.5, 0.5))
    objectness_score = 0.9
    words = _mock_words(size=geom[1], offset=geom[0])
    line = elements.Line(words)

    # Attribute checks
    assert len(line.words) == len(words)
    assert all(isinstance(w, elements.Word) for w in line.words)
    assert line.geometry == geom
    assert line.objectness_score == objectness_score

    # Render
    assert line.render() == "hello world"

    # Export
    assert line.export() == {
        "words": [w.export() for w in words],
        "geometry": geom,
        "objectness_score": objectness_score,
    }

    # Repr
    words_str = " " * 4 + ",\n    ".join(repr(word) for word in words) + ","
    assert line.__repr__() == f"Line(\n  (words): [\n{words_str}\n  ]\n)"

    # Ensure that words repr does't span on several lines when there are none
    assert repr(elements.Line([], ((0, 0), (1, 1)))) == "Line(\n  (words): []\n)"

    # from dict
    state_dict = {
        "words": [
            {
                "value": "there",
                "confidence": 0.1,
                "geometry": ((0, 0), (1.0, 1.0)),
                "objectness_score": objectness_score,
                "crop_orientation": {"value": 0, "confidence": None},
            }
        ],
        "geometry": ((0, 0), (1.0, 1.0)),
        "objectness_score": objectness_score,
    }
    line = elements.Line.from_dict(state_dict)
    assert line.export() == state_dict


def test_artefact():
    artefact_type = "qr_code"
    conf = 0.8
    geom = ((0, 0), (1, 1))
    artefact = elements.Artefact(artefact_type, conf, geom)

    # Attribute checks
    assert artefact.type == artefact_type
    assert artefact.confidence == conf
    assert artefact.geometry == geom

    # Render
    assert artefact.render() == "<[QR_CODE]>"

    # Export
    assert artefact.export() == {"type": artefact_type, "confidence": conf, "geometry": geom}

    # Repr
    assert artefact.__repr__() == f"Artefact(type='{artefact_type}', confidence={conf:.2})"


def test_layout_element():
    layout_type = "Title"
    conf = 0.9
    geom = ((0, 0), (1, 1))
    region = elements.LayoutElement(layout_type, conf, geom)

    # Attribute checks
    assert region.type == layout_type
    assert region.confidence == conf
    assert region.geometry == geom

    # Render
    assert region.render() == "<[TITLE]>"

    # Export
    assert region.export() == {"type": layout_type, "confidence": conf, "geometry": geom}

    # Repr
    assert region.__repr__() == f"LayoutElement(type='{layout_type}', confidence={conf:.2})"

    # Class method
    state_dict = {"geometry": ((0, 0), (0.5, 0.5)), "type": "Table", "confidence": 0.7}
    region = elements.LayoutElement.from_dict(state_dict)
    assert region.export() == state_dict


def test_table_cell():
    geom = ((0.1, 0.1), (0.3, 0.2))
    cell = elements.TableCell(
        value="hello", confidence=0.9, geometry=geom, row_start=0, row_end=1, col_start=2, col_end=2
    )

    # Attribute checks
    assert cell.value == "hello"
    assert cell.confidence == 0.9
    assert cell.geometry == geom
    assert (cell.row_start, cell.row_end, cell.col_start, cell.col_end) == (0, 1, 2, 2)
    assert cell.row_span == 2 and cell.col_span == 1

    # Render
    assert cell.render() == "hello"

    # Export
    assert cell.export() == {
        "geometry": geom,
        "value": "hello",
        "confidence": 0.9,
        "row_start": 0,
        "row_end": 1,
        "col_start": 2,
        "col_end": 2,
    }

    # Class method
    cell2 = elements.TableCell.from_dict(cell.export())
    assert cell2.export() == cell.export()


def _mock_table():
    # 2 x 2 table
    cells = [
        elements.TableCell("Name", 0.9, ((0.1, 0.1), (0.3, 0.2)), 0, 0, 0, 0),
        elements.TableCell("Age", 0.9, ((0.3, 0.1), (0.5, 0.2)), 0, 0, 1, 1),
        elements.TableCell("Alice", 0.9, ((0.1, 0.2), (0.3, 0.3)), 1, 1, 0, 0),
        elements.TableCell("30", 0.9, ((0.3, 0.2), (0.5, 0.3)), 1, 1, 1, 1),
    ]
    return elements.Table(cells=cells, num_rows=2, num_cols=2, geometry=((0.1, 0.1), (0.5, 0.3)), confidence=0.9)


def test_table():
    table = _mock_table()

    # Attribute checks
    assert table.num_rows == 2 and table.num_cols == 2
    assert len(table.cells) == 4
    assert all(isinstance(c, elements.TableCell) for c in table.cells)

    # Grid + render
    assert table.to_grid() == [["Name", "Age"], ["Alice", "30"]]
    assert table.render() == "Name\tAge\nAlice\t30"

    # Pandas
    df = pd.DataFrame(table.to_grid())
    assert df.shape == (2, 2)
    assert df.values.tolist() == [["Name", "Age"], ["Alice", "30"]]
    # With a header row
    table_grid = table.to_grid()
    df_h = pd.DataFrame(table_grid[1:], columns=table_grid[0])
    assert list(df_h.columns) == ["Name", "Age"]
    assert df_h.values.tolist() == [["Alice", "30"]]

    # Spanning cell: value placed at top-left of its span, the rest left empty
    spanned = elements.Table(
        cells=[
            elements.TableCell("merged", 0.9, ((0.0, 0.0), (1.0, 0.5)), 0, 0, 0, 1),
            elements.TableCell("a", 0.9, ((0.0, 0.5), (0.5, 1.0)), 1, 1, 0, 0),
            elements.TableCell("b", 0.9, ((0.5, 0.5), (1.0, 1.0)), 1, 1, 1, 1),
        ],
        num_rows=2,
        num_cols=2,
        geometry=((0.0, 0.0), (1.0, 1.0)),
    )
    assert spanned.to_grid() == [["merged", ""], ["a", "b"]]

    # Export
    exported = table.export()
    assert set(exported.keys()) == {"geometry", "num_rows", "num_cols", "confidence", "cells"}
    assert exported["cells"] == [c.export() for c in table.cells]

    # Class method round-trip
    assert elements.Table.from_dict(table.export()).export() == table.export()

    # Repr
    assert table.__repr__().startswith("Table(")


def test_prediction():
    prediction_str = "hello"
    conf = 0.8
    geom = ((0, 0), (1, 1))
    objectness_score = 0.9
    crop_orientation = {"value": 0, "confidence": None}
    prediction = elements.Prediction(prediction_str, conf, geom, objectness_score, crop_orientation)

    # Attribute checks
    assert prediction.value == prediction_str
    assert prediction.confidence == conf
    assert prediction.geometry == geom
    assert prediction.objectness_score == objectness_score
    assert prediction.crop_orientation == crop_orientation

    # Render
    assert prediction.render() == prediction_str

    # Export
    assert prediction.export() == {
        "value": prediction_str,
        "confidence": conf,
        "geometry": geom,
        "objectness_score": objectness_score,
        "crop_orientation": crop_orientation,
    }

    # Repr
    assert prediction.__repr__() == f"Prediction(value='hello', confidence={conf:.2}, bounding_box={geom})"

    # Class method
    state_dict = {
        "value": "there",
        "confidence": 0.1,
        "geometry": ((0, 0), (0.5, 0.5)),
        "objectness_score": 0.9,
        "crop_orientation": crop_orientation,
    }
    prediction = elements.Prediction.from_dict(state_dict)
    assert prediction.export() == state_dict


def test_block():
    geom = ((0, 0), (1, 1))
    sub_size = (geom[1][0] / 2, geom[1][0] / 2)
    objectness_score = 0.9
    lines = _mock_lines(size=sub_size, offset=geom[0])
    artefacts = _mock_artefacts(size=sub_size, offset=sub_size)
    block = elements.Block(lines, artefacts)

    # Attribute checks
    assert len(block.lines) == len(lines)
    assert len(block.artefacts) == len(artefacts)
    assert all(isinstance(w, elements.Line) for w in block.lines)
    assert all(isinstance(a, elements.Artefact) for a in block.artefacts)
    assert block.geometry == geom

    # Render
    assert block.render() == "hello world\nhello world"

    # Export
    assert block.export() == {
        "lines": [line.export() for line in lines],
        "artefacts": [artefact.export() for artefact in artefacts],
        "geometry": geom,
        "objectness_score": objectness_score,
    }


def test_page():
    page = np.zeros((300, 200, 3), dtype=np.uint8)
    page_idx = 0
    page_size = (300, 200)
    orientation = {"value": 0.0, "confidence": 0.0}
    language = {"value": "EN", "confidence": 0.8}
    blocks = _mock_blocks()
    layout = _mock_layout()
    page = elements.Page(page, blocks, page_idx, page_size, orientation, language, layout=layout)

    # Attribute checks
    assert len(page.blocks) == len(blocks)
    assert all(isinstance(b, elements.Block) for b in page.blocks)
    assert len(page.layout) == len(layout)
    assert all(isinstance(r, elements.LayoutElement) for r in page.layout)
    assert isinstance(page.page, np.ndarray)
    assert page.page_idx == page_idx
    assert page.dimensions == page_size
    assert page.orientation == orientation
    assert page.language == language

    # Render
    assert page.render() == "hello world\nhello world\n\nhello world\nhello world"

    # Export
    assert page.export() == {
        "blocks": [b.export() for b in blocks],
        "page_idx": page_idx,
        "dimensions": page_size,
        "orientation": orientation,
        "language": language,
        "layout": [r.export() for r in layout],
        "tables": [],
    }

    # Export XML
    xml_bytes, xml_tree = page.export_as_xml()
    assert isinstance(xml_bytes, (bytes, bytearray)) and isinstance(xml_tree, ElementTree)
    # The detected language must be exported instead of being hardcoded to "en"
    assert xml_tree.getroot().get("xml:lang") == "EN"
    # hOCR title properties must be single-spaced
    titles = [el.get("title") for el in xml_tree.iter() if el.get("title") is not None]
    assert any(title.startswith("bbox ") for title in titles)
    assert all("  " not in title for title in titles)
    # Without a detected language, the export must fall back to "en"
    fallback_page = elements.Page(np.zeros((300, 200, 3), dtype=np.uint8), blocks, page_idx, page_size, orientation)
    assert fallback_page.export_as_xml()[1].getroot().get("xml:lang") == "en"

    # Repr
    assert "\n".join(repr(page).split("\n")[:2]) == f"Page(\n  dimensions={page_size!r}"

    # Show
    page.show(block=False)

    # Synthesize
    img = page.synthesize()
    assert isinstance(img, np.ndarray)
    assert img.shape == (*page_size, 3)


def test_page_without_layout():
    # Backward compatibility: layout defaults to an empty list
    page = np.zeros((300, 200, 3), dtype=np.uint8)
    page = elements.Page(page, _mock_blocks(), 0, (300, 200))

    assert page.layout == []
    assert page.export()["layout"] == []


def test_kiepage():
    page = np.zeros((300, 200, 3), dtype=np.uint8)
    page_idx = 0
    page_size = (300, 200)
    orientation = {"value": 0.0, "confidence": 0.0}
    language = {"value": "EN", "confidence": 0.8}
    predictions = {CLASS_NAME: _mock_prediction()}
    layout = _mock_layout()
    kie_page = elements.KIEPage(page, predictions, page_idx, page_size, orientation, language, layout=layout)

    # Attribute checks
    assert len(kie_page.predictions) == len(predictions)
    assert all(isinstance(b, elements.Prediction) for b in kie_page.predictions[CLASS_NAME])
    assert len(kie_page.layout) == len(layout)
    assert all(isinstance(r, elements.LayoutElement) for r in kie_page.layout)
    assert isinstance(kie_page.page, np.ndarray)
    assert kie_page.page_idx == page_idx
    assert kie_page.dimensions == page_size
    assert kie_page.orientation == orientation
    assert kie_page.language == language

    # Render
    assert kie_page.render() == "words: hello\n\nwords: world"

    # Export
    assert kie_page.export() == {
        "predictions": {CLASS_NAME: [b.export() for b in predictions[CLASS_NAME]]},
        "page_idx": page_idx,
        "dimensions": page_size,
        "orientation": orientation,
        "language": language,
        "layout": [r.export() for r in layout],
    }

    # Export XML
    xml_bytes, xml_tree = kie_page.export_as_xml()
    assert isinstance(xml_bytes, (bytes, bytearray)) and isinstance(xml_tree, ElementTree)
    # The detected language must be exported instead of being hardcoded to "en"
    assert xml_tree.getroot().get("xml:lang") == "EN"
    # hOCR title properties must be single-spaced
    titles = [el.get("title") for el in xml_tree.iter() if el.get("title") is not None]
    assert any(title.startswith("bbox ") for title in titles)
    assert all("  " not in title for title in titles)
    # Without a detected language, the export must fall back to "en"
    fallback_page = elements.KIEPage(
        np.zeros((300, 200, 3), dtype=np.uint8), predictions, page_idx, page_size, orientation
    )
    assert fallback_page.export_as_xml()[1].getroot().get("xml:lang") == "en"

    # Repr
    assert "\n".join(repr(kie_page).split("\n")[:2]) == f"KIEPage(\n  dimensions={page_size!r}"

    # Show
    kie_page.show(block=False)

    # Synthesize
    img = kie_page.synthesize()
    assert isinstance(img, np.ndarray)
    assert img.shape == (*page_size, 3)


def test_document():
    pages = _mock_pages()
    doc = elements.Document(pages)

    # Attribute checks
    assert len(doc.pages) == len(pages)
    assert all(isinstance(p, elements.Page) for p in doc.pages)

    # Render
    page_export = "hello world\nhello world\n\nhello world\nhello world"
    assert doc.render() == f"{page_export}\n\n\n\n{page_export}"

    # Export
    assert doc.export() == {"pages": [p.export() for p in pages]}

    # Export XML
    xml_output = doc.export_as_xml()
    assert isinstance(xml_output, list) and len(xml_output) == len(pages)
    # Check that the XML is well-formed in hOCR format
    for xml_bytes, xml_tree in xml_output:
        assert isinstance(xml_bytes, bytes)
        assert isinstance(xml_tree, ElementTree)
        root = xml_tree.getroot()
        assert root.tag == "html"
        assert root[0].tag == "head"
        assert root[1].tag == "body"
        assert root[1][0].tag == "div" and root[1][0].attrib["class"] == "ocr_page"
        for block in root[1][0]:
            assert block.tag == "div" and block.attrib["class"] == "ocr_carea"
            assert block[0].tag == "p" and block[0].attrib["class"] == "ocr_par"
            for line in block[0]:
                assert line.tag == "span" and line.attrib["class"] == "ocr_line"
                for word in line:
                    assert word.tag == "span" and word.attrib["class"] == "ocrx_word"

    # Show
    doc.show(block=False)

    # Synthesize
    img_list = doc.synthesize()
    assert isinstance(img_list, list) and len(img_list) == len(pages)


def test_kie_document():
    pages = _mock_kie_pages()
    doc = elements.KIEDocument(pages)

    # Attribute checks
    assert len(doc.pages) == len(pages)
    assert all(isinstance(p, elements.KIEPage) for p in doc.pages)

    # Render
    page_export = "words: hello\n\nwords: world"
    assert doc.render() == f"{page_export}\n\n\n\n{page_export}"

    # Export
    assert doc.export() == {"pages": [p.export() for p in pages]}

    # Export XML
    xml_output = doc.export_as_xml()
    assert isinstance(xml_output, list) and len(xml_output) == len(pages)
    # Check that the XML is well-formed in hOCR format
    for xml_bytes, xml_tree in xml_output:
        assert isinstance(xml_bytes, bytes)
        assert isinstance(xml_tree, ElementTree)
        root = xml_tree.getroot()
        assert root.tag == "html"
        assert root[0].tag == "head"
        assert root[1].tag == "body"
        assert root[1][0].tag == "div" and root[1][0].attrib["class"] == "ocr_page"
        for block in root[1][0]:
            assert block.tag == "div" and block.attrib["class"] == "ocr_carea"
            assert block[0].tag == "p" and block[0].attrib["class"] == "ocr_par"
            for line in block[0]:
                assert line.tag == "span" and line.attrib["class"] == "ocr_line"
                for word in line:
                    assert word.tag == "span" and word.attrib["class"] == "ocrx_word"

    # Show
    doc.show(block=False)

    # Synthesize
    img_list = doc.synthesize()
    assert isinstance(img_list, list) and len(img_list) == len(pages)


def _word_at(text, x0, y0, x1, y1):
    return elements.Word(text, 0.95, ((x0, y0), (x1, y1)), 0.9, {"value": 0, "confidence": None})


def _line_at(text, x0, y0, x1, y1, rtl=False):
    """Build a line whose words are laid out geometrically (leftmost word = last logical word when rtl)"""
    words = text.split()
    step = (x1 - x0) / max(len(words), 1)
    geo_words = words[::-1] if rtl else words
    return elements.Line([
        _word_at(word, x0 + idx * step, y0, x0 + (idx + 0.9) * step, y1) for idx, word in enumerate(geo_words)
    ])


def _reading_order_page():
    """A page in the default builder configuration (single block) with a title, 2 columns & a footer"""
    lines = [_line_at("A Two Column Study", 0.2, 0.05, 0.8, 0.09)]
    lines += [_line_at(f"left line {idx}", 0.08, 0.14 + 0.05 * idx, 0.46, 0.17 + 0.05 * idx) for idx in range(3)]
    lines += [_line_at(f"right line {idx}", 0.54, 0.14 + 0.05 * idx, 0.92, 0.17 + 0.05 * idx) for idx in range(3)]
    lines += [_line_at("- item one", 0.08, 0.4, 0.46, 0.43), _line_at("Page 3 of 12", 0.4, 0.95, 0.6, 0.97)]
    # Shuffle the lines to make sure the export does not rely on the input order
    lines = [lines[idx] for idx in [5, 0, 8, 2, 4, 7, 1, 6, 3]]
    layout = [
        elements.LayoutElement("Title", 0.99, ((0.15, 0.04), (0.85, 0.1))),
        elements.LayoutElement("Text", 0.98, ((0.06, 0.12), (0.48, 0.32))),
        elements.LayoutElement("Text", 0.98, ((0.52, 0.12), (0.94, 0.32))),
        elements.LayoutElement("List-item", 0.97, ((0.06, 0.38), (0.48, 0.45))),
        elements.LayoutElement("Page-footer", 0.97, ((0.35, 0.94), (0.65, 0.98))),
    ]
    return elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8), [elements.Block(lines=lines)], 0, (1000, 800), layout=layout
    )


def test_page_items_in_reading_order():
    page = _reading_order_page()
    items = page.items_in_reading_order()
    assert all(isinstance(item, elements.Block) for item in items)
    rendered = [item.render(line_break=" ") for item in items]
    assert rendered[0] == "A Two Column Study"
    assert rendered[-1] == "Page 3 of 12"
    assert rendered.index("left line 0 left line 1 left line 2") < rendered.index(
        "right line 0 right line 1 right line 2"
    )
    # Multi-block pages are ordered at the block level
    top = elements.Block([_line_at("first words", 0.1, 0.1, 0.9, 0.15)])
    bottom = elements.Block([_line_at("last words", 0.1, 0.5, 0.9, 0.55)])
    page = elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [bottom, top], 0, (1000, 800))
    assert [block.render() for block in page.items_in_reading_order()] == ["first words", "last words"]


def test_page_export_as_markdown():
    page = _reading_order_page()
    markdown = page.export_as_markdown()
    parts = markdown.split("\n\n")
    assert parts[0] == "# A Two Column Study"
    assert parts[1] == "left line 0\nleft line 1\nleft line 2"
    # The list item belongs to the left column, hence it is read before the right column
    assert parts[2] == "- \\- item one"  # list item, with the raw OCR dash escaped
    assert parts[3] == "right line 0\nright line 1\nright line 2"
    assert parts[4] == "Page 3 of 12"
    # Page furniture can be dropped
    assert "Page 3 of 12" not in page.export_as_markdown(include_furniture=False)
    # Markdown structural characters are escaped by default
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [elements.Block([_line_at("*bold* #tag [link]", 0.1, 0.1, 0.9, 0.15)])],
        0,
        (1000, 800),
    )
    assert page.export_as_markdown() == "\\*bold\\* \\#tag \\[link\\]"
    assert page.export_as_markdown(escape=False) == "*bold* #tag [link]"
    # Empty pages export to an empty string
    assert elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [], 0, (1000, 800)).export_as_markdown() == ""


def test_page_export_as_markdown_rtl():
    # Two columns of Arabic text: the right column is read first, and the words of each line are emitted
    # from the rightmost to the leftmost one
    lines = [
        _line_at("النص في العمود الأيمن", 0.54, 0.1, 0.92, 0.14, rtl=True),
        _line_at("النص في العمود الأيسر", 0.08, 0.1, 0.46, 0.14, rtl=True),
    ]
    page = elements.Page(np.zeros((10, 10, 3), dtype=np.uint8), [elements.Block(lines=lines)], 0, (1000, 800))
    markdown = page.export_as_markdown()
    assert markdown == "النص في العمود الأيمن\n\nالنص في العمود الأيسر"
    # An explicit direction takes precedence over the detection
    assert page.export_as_markdown(direction="ltr").startswith("الأيسر")


def test_page_export_with_tables():
    cells = [
        elements.TableCell("Name", 0.9, ((0.1, 0.55), (0.4, 0.6)), 0, 0, 0, 0),
        elements.TableCell("Qty", 0.9, ((0.4, 0.55), (0.7, 0.6)), 0, 0, 1, 1),
        elements.TableCell("Bolt", 0.9, ((0.1, 0.6), (0.4, 0.65)), 1, 1, 0, 0),
        elements.TableCell("12|3", 0.9, ((0.4, 0.6), (0.7, 0.65)), 1, 1, 1, 1),
    ]
    table = elements.Table(cells, 2, 2, ((0.1, 0.55), (0.7, 0.65)), 0.95)
    lines = [
        _line_at("before the table", 0.1, 0.1, 0.9, 0.14),
        _line_at("after the table", 0.1, 0.7, 0.9, 0.74),
    ]
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8), [elements.Block(lines=lines)], 0, (1000, 800), tables=[table]
    )
    markdown = page.export_as_markdown()
    assert markdown.split("\n\n") == [
        "before the table",
        "| Name | Qty |\n| --- | --- |\n| Bolt | 12\\|3 |",
        "after the table",
    ]
    asciidoc = page.export_as_asciidoc()
    assert "|===\n|Name |Qty\n\n|Bolt |12\\|3\n|===" in asciidoc
    assert asciidoc.index("before the table") < asciidoc.index("|===") < asciidoc.index("after the table")


def test_page_export_as_asciidoc():
    page = _reading_order_page()
    asciidoc = page.export_as_asciidoc()
    parts = asciidoc.split("\n\n")
    assert parts[0] == "== A Two Column Study"
    assert parts[2] == "* {empty}- item one"
    assert "Page 3 of 12" not in page.export_as_asciidoc(include_furniture=False)


def test_page_export_as():
    page = _reading_order_page()
    assert page.export_as("markdown") == page.export_as("md") == page.export_as_markdown()
    assert page.export_as("adoc") == page.export_as_asciidoc()
    assert page.export_as("text") == page.render()
    assert page.export_as("json") == page.export()
    assert isinstance(page.export_as("xml")[0], bytes)
    assert page.export_as("markdown", include_furniture=False) == page.export_as_markdown(include_furniture=False)
    with pytest.raises(ValueError):
        page.export_as("yaml")


def test_document_export_as_markdown():
    pages = [
        elements.Page(
            np.zeros((10, 10, 3), dtype=np.uint8),
            [elements.Block([_line_at(f"page {idx} content", 0.1, 0.1, 0.9, 0.15)])],
            idx,
            (1000, 800),
        )
        for idx in range(2)
    ]
    doc = elements.Document(pages)
    assert doc.export_as_markdown() == "page 0 content\n\n---\n\npage 1 content"
    assert doc.export_as_asciidoc() == "page 0 content\n\n<<<\n\npage 1 content"
    assert doc.export_as_markdown(page_break="\n\n") == "page 0 content\n\npage 1 content"
    assert doc.export_as("markdown") == doc.export_as_markdown()
    assert doc.export_as("text") == doc.render()
    assert doc.export_as("json") == doc.export()
    assert len(doc.export_as("xml")) == 2
    with pytest.raises(ValueError):
        doc.export_as("pdf")


def test_kie_page_export_as_markdown():
    predictions = {
        CLASS_NAME: [
            elements.Prediction("second", 0.9, ((0.1, 0.5), (0.9, 0.6)), 0.9, {"value": 0, "confidence": None}),
            elements.Prediction("first", 0.9, ((0.1, 0.1), (0.9, 0.2)), 0.9, {"value": 0, "confidence": None}),
        ]
    }
    page = elements.KIEPage(np.zeros((10, 10, 3), dtype=np.uint8), predictions, 0, (1000, 800))
    assert page.export_as_markdown() == f"**{CLASS_NAME}**\n\n- first\n- second"
    assert page.export_as_asciidoc() == f"*{CLASS_NAME}*\n\n* first\n* second"
    assert page.export_as("md") == page.export_as_markdown()
    with pytest.raises(ValueError):
        page.export_as("yaml")
    doc = elements.KIEDocument([page])
    assert doc.export_as_markdown() == page.export_as_markdown()


def test_page_export_as_markdown_list_items():
    # Three list items spaced well apart (so the generic paragraph grouping keeps them separate) but each
    # covered by a List-item layout region: they must be coalesced into a single bulleted block.
    lines = [_line_at(f"item number {idx}", 0.1, 0.1 + 0.1 * idx, 0.5, 0.13 + 0.1 * idx) for idx in range(3)]
    layout = [
        elements.LayoutElement("List-item", 0.9, ((0.08, 0.09 + 0.1 * idx), (0.52, 0.14 + 0.1 * idx)))
        for idx in range(3)
    ]
    page = elements.Page(
        np.zeros((10, 10, 3), dtype=np.uint8), [elements.Block(lines=lines)], 0, (1000, 800), layout=layout
    )
    assert page.export_as_markdown() == "- item number 0\n- item number 1\n- item number 2"
    assert page.export_as_asciidoc() == "* item number 0\n* item number 1\n* item number 2"
    # the whole list is a single block in reading order
    items = page.items_in_reading_order()
    assert len(items) == 1
    assert len(items[0].lines) == 3
