"""Console entry `det-depth-distance-web`: serve the API with uvicorn."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from det_depth_distance.web.paths import WebPaths


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line and run the server (127.0.0.1 unless --host)."""
    parser = argparse.ArgumentParser(prog="det-depth-distance-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out-root", type=Path, default=Path("_testbed/outputs"))
    parser.add_argument("--library-root", type=Path, default=Path("_testbed/library"))
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        default=[],
        help="server folder videos may be imported from (repeatable)",
    )
    args = parser.parse_args(argv)

    import uvicorn

    from det_depth_distance.web.app import create_app

    paths = WebPaths(args.out_root, args.library_root, tuple(args.data_root))
    uvicorn.run(create_app(paths), host=args.host, port=args.port)
