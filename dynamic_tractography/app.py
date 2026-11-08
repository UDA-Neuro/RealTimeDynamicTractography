"""Local HTTP viewer for DYNAMIC TRACTOGRAPHY."""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import webbrowser

from .scene import build_scene
from .demo import demo_scene


class ViewerHandler(SimpleHTTPRequestHandler):
    scene: dict = {}

    def __init__(self, *args, **kwargs):
        web_root = Path(__file__).with_name("web")
        super().__init__(*args, directory=str(web_root), **kwargs)

    def do_GET(self):
        if self.path == "/scene.json":
            payload = json.dumps(self.scene, separators=(",", ":")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            super().do_GET()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DYNAMIC TRACTOGRAPHY research viewer")
    parser.add_argument("--subject", type=Path, help="FreeSurfer recon-all subject directory")
    parser.add_argument("--electrodes", type=Path,
                        help="CSV: channel_number,label,x,y,z in T1 scanner RAS millimetres")
    parser.add_argument("--spike", type=Path, help="RealTimeSpike CSV")
    parser.add_argument("--hfo", type=Path, help="RealTimeHFO CSV")
    parser.add_argument("--tracts", type=Path, help="DSI Studio exported .trk or .trk.gz")
    parser.add_argument("--tract-transform", type=Path,
                        help="Optional 4 x 4 text affine from tract RAS to T1 scanner RAS")
    parser.add_argument("--source-label", default="PATIENT TRACTOGRAPHY")
    parser.add_argument("--min-delay-ms", type=float, default=10)
    parser.add_argument("--max-delay-ms", type=float, default=50)
    parser.add_argument("--electrode-radius-mm", type=float, default=8)
    parser.add_argument("--cross-type", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--export-scene", type=Path, help="Write scene JSON before serving")
    parser.add_argument("--demo", action="store_true", help="Preview synthetic data")
    args = parser.parse_args(argv)
    if args.demo:
        scene = demo_scene()
    else:
        if not args.electrodes or (not args.spike and not args.hfo):
            parser.error("Provide --electrodes and at least one of --spike or --hfo")
        if args.source_label.upper().startswith("HCP") and not args.tract_transform:
            parser.error("HCP template tracks need a verified transform into patient T1 space")
        scene = build_scene(args.subject, args.electrodes, args.spike, args.hfo,
                            args.tracts, args.tract_transform, args.min_delay_ms,
                            args.max_delay_ms, args.electrode_radius_mm,
                            args.cross_type, args.source_label)
    if args.export_scene:
        args.export_scene.write_text(json.dumps(scene), encoding="utf-8")
    ViewerHandler.scene = scene
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"DYNAMIC TRACTOGRAPHY  {url}")
    print(f"{len(scene['events'])} events | {len(scene['propagations'])} propagation candidates")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

