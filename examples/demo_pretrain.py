import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Pretraining demo: watch loss drop and skill rise."""
from panguforge.config import load_config
from panguforge.backends import get_backend
from panguforge.data import SyntheticDataset
from panguforge.train import PretrainLoop


def main():
    cfg = load_config("config.example.yaml")
    eng = get_backend(cfg.get("backend", "mock"), cfg)
    ds = SyntheticDataset(vocab_size=int(cfg.get("model", {}).get("vocab_size", 512)), seed=0)
    m = PretrainLoop(eng, cfg).run(ds)
    print("pretrain result:", m)


if __name__ == "__main__":
    main()
