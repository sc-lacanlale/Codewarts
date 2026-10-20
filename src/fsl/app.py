"""Cross-platform entry point: ``python -m fsl.app``.

``--smoke`` loads the LSTM and scores a synthetic landmark window. It does not
open a camera or a window. The desktop UI is imported only for the webcam modes
so this command stays usable in CI.
"""

from __future__ import annotations

import argparse
import sys

from fsl.model import ModelName, format_smoke_report, smoke_inference


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fsl.app",
        description="Filipino Sign Language translator (MediaPipe landmarks, LSTM, Tagalog TTS).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Load the model and run inference on a 30x258 zero sequence. No webcam.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Open the scripted demo UI instead of live recognition.",
    )
    parser.add_argument(
        "--model",
        choices=("fsl15", "fsl11"),
        default="fsl15",
        help="Which committed checkpoint --smoke should load. Default: fsl15.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.smoke and args.demo:
        parser.error("--smoke and --demo cannot be combined")
    if args.smoke:
        model_name: ModelName = args.model
        try:
            report = smoke_inference(model_name)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(format_smoke_report(report))
        return 0
    if args.demo:
        from apps.demo import main as demo_main

        demo_main()
        return 0
    from apps.desktop import main as desktop_main

    desktop_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
