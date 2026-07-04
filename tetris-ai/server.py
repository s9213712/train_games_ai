from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from rl_trainer import RLTrainer


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
trainer = RLTrainer(ROOT)


class TetrisHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self) -> None:
        if self.path == "/api/rl/state":
            self._json(trainer.snapshot())
            return
        if self.path == "/api/rl/replay/latest":
            self._json(trainer.latest_replay_payload())
            return
        self._static()

    def do_POST(self) -> None:
        payload = self._read_json()
        if self.path == "/api/rl/start":
            trainer.update_config(payload)
            trainer.start()
            self._json(trainer.snapshot())
            return
        if self.path == "/api/rl/pause":
            trainer.pause()
            self._json(trainer.snapshot())
            return
        if self.path == "/api/rl/reset":
            trainer.pause()
            trainer.reset()
            self._json(trainer.snapshot())
            return
        if self.path == "/api/rl/config":
            trainer.update_config(payload)
            self._json(trainer.snapshot())
            return
        if self.path == "/api/rl/step":
            episodes = max(1, min(500, int(payload.get("episodes", 1) or 1)))
            trainer.update_config(payload)
            trainer.pause()
            result = trainer.train_guarded_batch(
                episodes,
                eval_episodes=max(2, min(24, int(payload.get("eval_episodes", 2) or 2))),
                accept_ratio=max(0.7, min(1.1, float(payload.get("accept_ratio", 0.98) or 0.98))),
            )
            data = trainer.snapshot()
            data["step_result"] = result
            self._json(data)
            return
        if self.path == "/api/rl/evaluate":
            episodes = max(1, min(200, int(payload.get("episodes", 20) or 20)))
            trainer.pause()
            include_hold = payload.get("lookahead_include_hold")
            self._json({"evaluation": trainer.evaluate(episodes, lookahead_include_hold=include_hold if include_hold is not None else None), "state": trainer.snapshot()})
            return
        self.send_error(404)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    def _json(self, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _static(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            path = "/index.html"
        target = (WEB_ROOT / path.lstrip("/")).resolve()
        if not str(target).startswith(str(WEB_ROOT.resolve())) or not target.exists() or not target.is_file():
            self.send_error(404)
            return
        content = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7871)
    args = parser.parse_args()
    try:
        with ThreadingHTTPServer((args.host, args.port), TetrisHandler) as server:
            server.serve_forever()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
