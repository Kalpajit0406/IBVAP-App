"""
test_evidence.py -- EvidenceChain SHA-256 hash chain: append, verify, and
tamper detection. No model, no GPU.

    python tests/test_evidence.py
    pytest  tests/test_evidence.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibvap.evidence import EvidenceChain                     # noqa: E402


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


def _fresh() -> Path:
    return Path(tempfile.mkdtemp()) / "chain.jsonl"


@case("a fresh chain verifies clean")
def _():
    ec = EvidenceChain(str(_fresh()))
    ec.append({"cam_id": 0, "level": "High"})
    ec.append({"cam_id": 1, "level": "Critical"})
    ok, bad = ec.verify()
    assert ok and bad == -1, (ok, bad)


@case("prev_hash links each record to the one before it")
def _():
    p = _fresh()
    ec = EvidenceChain(str(p))
    h1 = ec.append({"a": 1})
    h2 = ec.append({"a": 2})
    lines = p.read_text().splitlines()
    r1, r2 = json.loads(lines[0]), json.loads(lines[1])
    assert r1["hash"] == h1
    assert r2["prev_hash"] == h1
    assert r2["hash"] == h2


@case("reopening the chain continues from the tip")
def _():
    p = _fresh()
    EvidenceChain(str(p)).append({"a": 1})
    ec2 = EvidenceChain(str(p))          # re-load tip from disk
    ec2.append({"a": 2})
    assert ec2.verify()[0]
    assert len(p.read_text().splitlines()) == 2


@case("editing a record's payload breaks verification at that line")
def _():
    p = _fresh()
    ec = EvidenceChain(str(p))
    ec.append({"score": 10})
    ec.append({"score": 20})
    ec.append({"score": 30})
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["event"]["score"] = 999                       # tamper, keep the old hash
    lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    ok, bad = EvidenceChain(str(p)).verify()
    assert not ok and bad == 2, (ok, bad)


@case("deleting a record breaks the chain")
def _():
    p = _fresh()
    ec = EvidenceChain(str(p))
    for i in range(4):
        ec.append({"i": i})
    lines = p.read_text().splitlines()
    del lines[1]
    p.write_text("\n".join(lines) + "\n")
    ok, _bad = EvidenceChain(str(p)).verify()
    assert not ok


@case("a half-written trailing record (live append) is not reported as tampering")
def _():
    p = _fresh()
    ec = EvidenceChain(str(p))
    ec.append({"i": 1})
    ec.append({"i": 2})
    with p.open("ab") as f:
        f.write(b'{"timestamp": 1.0, "prev_ha')      # append() mid-write
    # A reader that doesn't re-open for writing (as /api/evidence/verify does)
    probe = EvidenceChain.__new__(EvidenceChain)
    probe._path = p
    ok, bad = probe.verify()
    assert ok and bad == -1, (ok, bad)
    s = probe.summary()
    assert s["records"] == 2 and s["ok"]


@case("a garbage line reports failure at that line instead of raising")
def _():
    p = _fresh()
    ec = EvidenceChain(str(p))
    ec.append({"i": 1})
    with p.open("a") as f:
        f.write("not json at all\n")
    ok, bad = EvidenceChain(str(p)).verify()
    assert not ok and bad == 2, (ok, bad)


@case("reopening after a crash fragment keeps new records on their own lines")
def _():
    p = _fresh()
    ec = EvidenceChain(str(p))
    h1 = ec.append({"i": 1})
    with p.open("ab") as f:
        f.write(b'{"partial')
    ec2 = EvidenceChain(str(p))
    ec2.append({"i": 2})
    lines = p.read_text().splitlines()
    assert len(lines) == 3, lines
    assert json.loads(lines[2])["prev_hash"] == h1
    ok, bad = ec2.verify()
    assert not ok and bad == 2, (ok, bad)     # the fragment stays visible


@case("tail() is newest-first and summary() reports count + tip")
def _():
    p = _fresh()
    ec = EvidenceChain(str(p))
    hs = [ec.append({"i": i}) for i in range(5)]
    t = ec.tail(3)
    assert [r["event"]["i"] for r in t] == [4, 3, 2]
    s = ec.summary()
    assert s["records"] == 5 and s["tip"] == hs[-1] and s["ok"]


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
