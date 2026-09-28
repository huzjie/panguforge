import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""RL (GRPO) demo: reward increases as skill rises."""
from panguforge.config import load_config
from panguforge.backends import get_backend
from panguforge.data import InstructionDataset
from panguforge.train import RLLoop


def main():
    cfg = load_config("config.example.yaml")
    eng = get_backend(cfg.get("backend", "mock"), cfg)
    m = RLLoop(eng, cfg).run(InstructionDataset(seed=0))
    print("rl result:", m)


if __name__ == "__main__":
    main()
