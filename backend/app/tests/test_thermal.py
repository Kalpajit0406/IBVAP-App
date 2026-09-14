"""
test_thermal.py -- train_thermal.py label conversion + assembly. No GPU / model.

    python tests/test_thermal.py
    pytest  tests/test_thermal.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                    # noqa: E402
import cv2                                            # noqa: E402
from training import train_thermal as tt                       # noqa: E402

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


_LLVIP_XML = """<annotation>
  <size><width>1280</width><height>1024</height></size>
  <object><name>person</name>
    <bndbox><xmin>640</xmin><ymin>512</ymin><xmax>768</xmax><ymax>768</ymax></bndbox></object>
  <object><name>traffic light</name>
    <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>20</xmax><ymax>20</ymax></bndbox></object>
</annotation>"""


@case("_voc_to_yolo: person -> COCO id 0, normalised centre/size; drops others")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.xml"
        p.write_text(_LLVIP_XML)
        lines = tt._voc_to_yolo(p)
        assert lines == ["0 0.550000 0.625000 0.100000 0.250000"], lines


@case("_voc_to_yolo: falls back to img_wh when <size> is absent")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "b.xml"
        p.write_text("<annotation><object><name>person</name><bndbox>"
                     "<xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>100</ymax>"
                     "</bndbox></object></annotation>")
        lines = tt._voc_to_yolo(p, img_wh=(100, 200))
        assert lines == ["0 0.250000 0.250000 0.500000 0.500000"], lines


@case("NAME2COCO: the vehicle synonyms map to the right COCO ids")
def _():
    assert tt.NAME2COCO["bike"] == 1 and tt.NAME2COCO["bicycle"] == 1
    assert tt.NAME2COCO["motor"] == 3 and tt.NAME2COCO["motorcycle"] == 3
    assert tt.NAME2COCO["car"] == 2 and tt.NAME2COCO["bus"] == 5 and tt.NAME2COCO["truck"] == 7
    assert "dog" not in tt.NAME2COCO and "traffic light" not in tt.NAME2COCO


@case("_remap_yolo_lines: Roboflow class indices -> COCO ids, unmapped dropped")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.txt"
        # names=['bike','car','dog','person'] -> idx2coco={0:1,1:2,3:0}
        p.write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.3 0.3\n2 0.9 0.9 0.1 0.1\n3 0.4 0.4 0.2 0.2\n")
        idx2coco = {0: 1, 1: 2, 3: 0}
        out = tt._remap_yolo_lines(p, idx2coco)
        assert out == ["1 0.5 0.5 0.1 0.1", "2 0.2 0.2 0.3 0.3", "0 0.4 0.4 0.2 0.2"], out


@case("assemble: LLVIP 19* -> val, else train; labels converted; data.yaml nc:80")
def _():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "ds"
        inf = src / "llvip" / "infrared"
        ann = src / "llvip" / "Annotations"
        inf.mkdir(parents=True); ann.mkdir(parents=True)
        blank = np.zeros((32, 40, 3), np.uint8)
        for stem in ("010001", "010002", "190001"):
            cv2.imwrite(str(inf / f"{stem}.jpg"), blank)
            (ann / f"{stem}.xml").write_text(_LLVIP_XML)
        out = src / "thermal_yolo"
        counts = tt.assemble(src, out, max_per_source=100)
        assert counts["llvip_train"] == 2 and counts["llvip_val"] == 1, counts
        assert counts["per_class"].get(0) == 3, counts          # one person box each
        y = tt.write_data_yaml(out)
        import yaml
        loaded = yaml.safe_load(y.read_text())
        assert loaded["nc"] == 80 and loaded["names"][0] == "person"
        # a converted label file exists and starts with class 0
        lbls = list((out / "labels" / "train").glob("*.txt"))
        assert lbls and lbls[0].read_text().startswith("0 ")


@case("assemble: re-run wipes the previous output (no accumulation)")
def _():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "ds"
        inf = src / "llvip" / "infrared"; ann = src / "llvip" / "Annotations"
        inf.mkdir(parents=True); ann.mkdir(parents=True)
        blank = np.zeros((16, 16, 3), np.uint8)
        for stem in ("010001", "190001"):
            cv2.imwrite(str(inf / f"{stem}.jpg"), blank)
            (ann / f"{stem}.xml").write_text(_LLVIP_XML)
        out = src / "thermal_yolo"
        tt.assemble(src, out, max_per_source=100)
        tt.assemble(src, out, max_per_source=100)
        assert len(list((out / "images" / "train").glob("*.jpg"))) == 1


def run() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  -- {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(run())
