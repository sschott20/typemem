#!/usr/bin/env python3
"""Start the typemem visualizer."""
import argparse
import signal
import sys

from typemem.memory_manager import MemoryManager
from typemem.viz import start_viz


def main():
    parser = argparse.ArgumentParser(description="typemem visualizer")
    parser.add_argument("db_path", nargs="?",
                        default="/home/gc635/Documents/TypeGo/src/typego/resource/memory_db",
                        help="Path to ChromaDB directory (default: TypeGo)")
    parser.add_argument("-p", "--port", type=int, default=8811)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    manager = MemoryManager(persist_dir=args.db_path, robot_id="viz")
    server = start_viz(manager, port=args.port, open_browser=not args.no_browser)
    print(f"typemem viz running at http://localhost:{server.port}")
    print(f"Database: {args.db_path}")
    print("Ctrl+C to stop")

    try:
        signal.pause()
    except KeyboardInterrupt:
        server.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
