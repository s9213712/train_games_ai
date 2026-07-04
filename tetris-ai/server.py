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
WEB_ROOT_RESOLVED = WEB_ROOT.resolve()
MAX_JSON_BODY_BYTES = 64 * 1024
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
        try:
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
                    accept_min_delta=max(-100000.0, min(100000.0, float(payload.get("accept_min_delta", 0.0) or 0.0))),
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
        except (TypeError, ValueError) as exc:
            self._json({"ok": False, "error": str(exc)}, status=400)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        if length > MAX_JSON_BODY_BYTES:
            raise ValueError("JSON request body is too large")
        if length <= 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON request body") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON request body must be an object")
        return payload

    def _json(self, payload: dict, *, status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
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
        try:
            target.relative_to(WEB_ROOT_RESOLVED)
        except ValueError:
            self.send_error(404)
            return
        if not target.exists() or not target.is_file():
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
