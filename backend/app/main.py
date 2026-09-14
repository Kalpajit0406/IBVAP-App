"""
IBVAP — Multi-stream demo entry point.

Usage:
    python main.py                    # uses config.yaml
    python main.py --config my.yaml   # custom config
    python main.py --verify-chain     # verify evidence chain integrity and exit
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import yaml

from ibvap.detector import Detector
from ibvap.display import GridDisplay
from ibvap.event_store import EventStore
from ibvap.evidence import EvidenceChain
from ibvap.risk_engine import RiskAssessment, RiskEngine
from ibvap.rtsp_capture import RtspCapture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ibvap.main")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="IBVAP demo pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verify-chain", action="store_true",
                        help="Verify evidence hash chain integrity and exit")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.verify_chain:
        chain = EvidenceChain(cfg["evidence"]["hash_chain_path"])
        ok, bad_line = chain.verify()
        if ok:
            print("Chain OK — no tampering detected.")
        else:
            print(f"Chain BROKEN at line {bad_line}!")
            sys.exit(1)
        return

    # Local OpenCV mode only pulls real sources — skip phone (ws) slots and
    # anything marked disabled. Use server.py for the phone / web pipeline.
    _WS = {"", "ws", "phone", "mobile", "browser"}
    streams_cfg = [s for s in cfg["streams"]
                   if s.get("enabled", True)
                   and str(s.get("url", "")).strip().lower() not in _WS]
    if not streams_cfg:
        print("No pullable streams in config.yaml (all are phone/ws slots or "
              "disabled).\nAdd rtsp:// / webcam / file entries, or run "
              "'python server.py' for the phone pipeline.")
        return
    log_from = cfg["evidence"].get("log_level", "High")

    # ── Start capture threads ─────────────────────────────────────────────────
    # Same RtspCapture the web server uses: one decode thread per source, frames
    # normalised to a single size, auto-reconnect + stall watchdog. `cctv:` /
    # `ingest:` config blocks apply here too.
    ing = cfg.get("ingest", {}) or {}
    c = cfg.get("cctv", {}) or {}
    norm_w = int(ing.get("normalise_width", 1280))
    norm_h = int(ing.get("normalise_height", 720))
    captures = [
        RtspCapture(
            s["id"], str(s["url"]),
            name=s.get("name"),
            norm_w=norm_w, norm_h=norm_h,
            transport=str(s.get("transport", c.get("transport", "tcp"))),
            decode_fps=float(s.get("decode_fps", c.get("decode_fps", 15))),
            reconnect_delay=float(c.get("reconnect_delay", 3.0)),
            stall_timeout=float(c.get("stall_timeout", 8.0)),
            open_timeout=float(c.get("open_timeout", 8.0)),
        ).start()
        for s in streams_cfg
    ]

    detector = Detector(cfg, num_cameras=len(captures))
    risk_engine = RiskEngine(cfg)
    evidence = EvidenceChain(cfg["evidence"]["hash_chain_path"])
    store = EventStore(cfg["evidence"]["db_path"])
    display = GridDisplay(cfg)

    prev_levels: dict[int, str] = {}
    log_levels = {"High", "Critical"} if log_from == "High" else {"Critical"}

    logger.info("IBVAP running — %d stream(s). Press 'q' to quit.", len(captures))

    try:
        while True:
            results = []
            for cap in captures:
                frame_data = cap.read()
                if frame_data is None:
                    continue
                ts, frame = frame_data
                sr = detector.process(cap.cam_id, ts, frame)
                results.append(sr)

            risks: dict[int, RiskAssessment] = {}
            for sr in results:
                ra = risk_engine.assess(sr)
                risks[sr.cam_id] = ra

                # Log only on level transitions to avoid flooding
                prev = prev_levels.get(sr.cam_id, "Normal")
                if ra.level != prev:
                    prev_levels[sr.cam_id] = ra.level
                    if ra.level in log_levels:
                        event = {
                            "cam_id": sr.cam_id,
                            "level": ra.level,
                            "score": round(ra.score, 2),
                            "persons": sr.person_count,
                            "vehicles": sr.vehicle_count,
                        }
                        ev_hash = evidence.append(event)
                        store.log(
                            cam_id=sr.cam_id,
                            level=ra.level,
                            score=ra.score,
                            persons=sr.person_count,
                            vehicles=sr.vehicle_count,
                            details=event,
                            ev_hash=ev_hash,
                        )
                        logger.warning(
                            "CAM-%02d %s alert — score=%.1f P=%d V=%d hash=%s…",
                            sr.cam_id, ra.level, ra.score,
                            sr.person_count, sr.vehicle_count,
                            ev_hash[:12],
                        )

            if not display.show(results, risks):
                break

    finally:
        logger.info("Shutting down…")
        for cap in captures:
            cap.stop()
        display.close()
        logger.info("Done.")


if __name__ == "__main__":
    main()
