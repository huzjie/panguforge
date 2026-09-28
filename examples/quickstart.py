import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""Quickstart: build config -> pretrain -> sft -> rl -> generate, all in one."""
from panguforge.config import load_config
from panguforge.backends import get_backend
from panguforge.data import SyntheticDataset, InstructionDataset
from panguforge.train import PretrainLoop, SFTLoop, RLLoop


def main():
    cfg = load_config("config.example.yaml", defaults={
        "model": {"vocab_size": 512, "dim": 64, "num_heads": 4, "num_layers": 2,
                  "moe_hidden": 256, "num_experts": 8, "top_k": 2, "max_seq": 512, "seed": 0},
        "backend": "mock",
        "mock": {"skill_init": 0.4, "lr": 0.05},
        "train": {"steps": 60},
        "rl": {"steps": 20, "algorithm": "grpo"},
    })
    eng = get_backend(cfg["backend"], cfg)
    print(f"initial skill = {eng.skill}")

    pretrain = PretrainLoop(eng, cfg).run(SyntheticDataset(vocab_size=512, seed=0))
    sft = SFTLoop(eng, cfg).run(InstructionDataset(seed=0))
    rl = RLLoop(eng, cfg).run(InstructionDataset(seed=0))
    print(f"final skill = {eng.skill}")
    print("sample generation:", eng.generate("今天天气", max_tokens=16))


if __name__ == "__main__":
    main()
