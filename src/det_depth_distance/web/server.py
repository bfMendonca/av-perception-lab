"""Console entry `det-depth-distance-web`: serve the API with uvicorn."""

import argparse
import sys
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
        help="server folder videos may be imported from (repeatable; default: data)",
    )
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=None,
        help="built front-end to serve at / (default: web/dist of the checkout)",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="extra Host name the server answers to, e.g. the LAN address or "
        "name used in the browser (repeatable; loopback and --host always are)",
    )
    parser.add_argument(
        "--dev-origin",
        action="append",
        default=[],
        help="extra origin allowed to send POST/PUT/PATCH/DELETE, e.g. "
        "http://localhost:5173 for a dev server (repeatable)",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        action="append",
        default=[],
        help="folder the path fields of a run config (weights, labels, "
        "calibration, fonts...) must resolve inside (repeatable; default: "
        "anywhere outside the library and out-root)",
    )
    args = parser.parse_args(argv)

    import uvicorn

    from det_depth_distance.web import security
    from det_depth_distance.web.app import create_app

    data_roots = tuple(args.data_root) or (Path("data"),)
    paths = WebPaths(
        args.out_root, args.library_root, data_roots, tuple(args.asset_root)
    )
    for warning in startup_warnings(args.host, args.allowed_host, args.asset_root):
        print(f"warning: {warning}", file=sys.stderr, flush=True)
    extra = {} if args.static_dir is None else {"static_dir": args.static_dir}
    app = create_app(
        paths,
        allowed_hosts=security.allowed_hosts(args.host, args.allowed_host),
        dev_origins=args.dev_origin,
        **extra,
    )
    # Open SSE streams never end by themselves: after 3 s uvicorn cancels
    # them and still runs the lifespan shutdown (which ends the jobs).
    uvicorn.run(app, host=args.host, port=args.port, timeout_graceful_shutdown=3)


def startup_warnings(
    host: str, allowed: Sequence[str], asset_roots: Sequence[Path]
) -> list[str]:
    """What to tell the owner when the server is reachable from the network."""
    from det_depth_distance.web import security

    if security.is_loopback(host):
        return []
    out = []
    if not asset_roots:
        out.append(
            f"--host {host} without --asset-root: any client on the network may "
            "point a run's weights/labels/repo at any server file outside the "
            "library and out-root (pass --asset-root for each asset folder)"
        )
    if host in security.WILDCARD_HOSTS and not allowed:
        out.append(
            f"--host {host} without --allowed-host: only http://127.0.0.1 and "
            "http://localhost are answered; add --allowed-host with the address "
            "or name other machines use"
        )
    return out
