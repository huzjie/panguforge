import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Inference demo across backends."""
from panguforge.config import load_config
from panguforge.backends import get_backend, list_backends


def main():
    cfg = load_config("config.example.yaml")
    print("available backends:", list_backends())
    for name in ["mock", "cpu"]:
        try:
            eng = get_backend(name, cfg)
            out = eng.generate("你好，", max_tokens=16)
            print(f"{name}: {out}")
        except Exception as e:
            print(f"{name}: skipped ({e})")


if __name__ == "__main__":
    main()
