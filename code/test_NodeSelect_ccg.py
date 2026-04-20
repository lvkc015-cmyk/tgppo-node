import test_NodeSelect as base
from NodeSelect.actor_ccg import ActorCCG
from NodeSelect.critic_ccg import CriticCCG

# Codex note: keep the original testing script untouched and swap in the CCG architecture here.
base.Actor = ActorCCG
base.Critic = CriticCCG


if __name__ == "__main__":
    base.main()
