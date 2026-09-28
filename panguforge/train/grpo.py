"""Group Relative Policy Optimization (GRPO).

For each prompt sample a group of K completions, score them, and update the
policy toward the higher-reward completions using a relative advantage.
"""
import math


def grpo_update(engine, prompts, group_size=4, lr=0.05):
    """Run one GRPO iteration. Returns metrics dict."""
    rewards_all = []
    for p in prompts:
        group = [engine.generate(p, max_tokens=16, sample=True) for _ in range(group_size)]
        rs = [engine.reward(p, g) for g in group]
        rewards_all.append((p, group, rs))
    # advantage = reward - mean(reward) per prompt group
    total_adv = 0.0
    n = 0
    for p, group, rs in rewards_all:
        mean = sum(rs) / len(rs)
        for g, r in zip(group, rs):
            adv = r - mean
            total_adv += adv
            n += 1
            if adv > 0:
                engine.reinforce(p, g, lr * adv)
    avg_reward = sum(r for _, _, rs in rewards_all for r in rs) / max(1, n)
    return {"reward": avg_reward, "skill": engine.skill, "mean_advantage": total_adv / max(1, n)}
