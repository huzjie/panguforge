import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Start the HTTP server programmatically."""
from panguforge.config import load_config
from panguforge.backends import get_backend
from panguforge.serving.server import create_server


def main():
    cfg = load_config("config.example.yaml")
    eng = get_backend(cfg.get("backend", "mock"), cfg)
    server = create_server(eng, "127.0.0.1", 8000)
    print("serving on http://127.0.0.1:8000 (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
