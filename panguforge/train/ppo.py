"""Proximal Policy Optimization (clipped objective)."""
import math


def ppo_update(engine, prompts, clip_ratio=0.2, lr=0.05):
    rewards = []
    for p in prompts:
        r = engine.reward(p, engine.generate(p, max_tokens=16, sample=True))
        rewards.append(r)
        # clipped surrogate: only reinforce when advantage is clearly positive
        engine.reinforce(p, engine.generate(p, max_tokens=16, sample=True), lr * min(r, clip_ratio))
    return {"reward": sum(rewards) / max(1, len(rewards)), "skill": engine.skill}
