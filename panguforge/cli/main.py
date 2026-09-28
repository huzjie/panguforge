"""Command-line entry point."""
import argparse
import json
import sys

from ..config import load_config
from ..logging import get_logger
from ..backends import get_backend, list_backends


def cmd_doctor(args):
    cfg = load_config(args.config)
    logger = get_logger("doctor")
    logger.info("panguforge doctor")
    for name in list_backends():
        try:
            eng = get_backend(name, cfg)
            extra = ""
            if hasattr(eng, "status"):
                extra = str(eng.status())
            logger.info(f"  backend {name}: OK {extra}")
        except Exception as e:
            logger.warning(f"  backend {name}: {e}")
    return {"status": "ok"}


def cmd_pretrain(args):
    from ..backends import get_backend
    from ..data import SyntheticDataset
    from ..train import PretrainLoop
    cfg = load_config(args.config)
    logger = get_logger("pretrain")
    eng = get_backend(cfg.get("backend", "mock"), cfg)
    ds = SyntheticDataset(vocab_size=int(cfg.get("model", {}).get("vocab_size", 512)), seed=0)
    m = PretrainLoop(eng, cfg, logger=logger).run(ds)
    print(json.dumps(m, ensure_ascii=False))
    return m


def cmd_sft(args):
    from ..data import InstructionDataset
    from ..train import SFTLoop
    cfg = load_config(args.config)
    logger = get_logger("sft")
    eng = get_backend(cfg.get("backend", "mock"), cfg)
    ds = InstructionDataset(seed=0)
    m = SFTLoop(eng, cfg, logger=logger).run(ds)
    print(json.dumps(m, ensure_ascii=False))
    return m


def cmd_rl(args):
    from ..data import InstructionDataset
    from ..train import RLLoop
    cfg = load_config(args.config)
    logger = get_logger("rl")
    eng = get_backend(cfg.get("backend", "mock"), cfg)
    ds = InstructionDataset(seed=0)
    m = RLLoop(eng, cfg, logger=logger).run(ds)
    print(json.dumps(m, ensure_ascii=False))
    return m


def cmd_serve(args):
    from ..serving.server import create_server
    cfg = load_config(args.config)
    logger = get_logger("serve")
    eng = get_backend(cfg.get("backend", "mock"), cfg)
    host = args.host
    port = int(args.port)
    server = create_server(eng, host, port)
    logger.info(f"serving on http://{host}:{port} (backend={eng.name})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_bench(args):
    from ..benchmark.benchmarks import run_bench
    cfg = load_config(args.config)
    results = run_bench(cfg)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml", help="config file (yaml/json)")

    p = argparse.ArgumentParser(prog="panguforge", description="openPangu-2.0 style MoE LLM training & inference framework")
    p.add_argument("--config", default="config.yaml", help="config file (yaml/json)")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("doctor", parents=[common], help="environment & backend check")
    sub.add_parser("pretrain", parents=[common], help="language-modeling pretraining")
    sub.add_parser("sft", parents=[common], help="supervised fine-tuning")
    sub.add_parser("rl", parents=[common], help="RL post-training (GRPO/PPO)")
    sp = sub.add_parser("serve", parents=[common], help="start HTTP inference server")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", default=8000)
    sub.add_parser("bench", parents=[common], help="run benchmarks")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    handlers = {
        "doctor": cmd_doctor,
        "pretrain": cmd_pretrain,
        "sft": cmd_sft,
        "rl": cmd_rl,
        "serve": cmd_serve,
        "bench": cmd_bench,
    }
    fn = handlers.get(args.command)
    if fn is None:
        build_parser().print_help()
        return 1
    fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
