#!/usr/bin/env python3
"""Compatibility entry point for the current CPU-only website browser audit."""
import argparse
from http.server import SimpleHTTPRequestHandler
from pathlib import Path


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def copyfile(self, source, outputfile):
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    from verify_website_editorial import verify
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chromium', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1]/'docs')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    verify(args.site, args.output, args.chromium, args.quick)
