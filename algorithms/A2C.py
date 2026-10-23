from QAC import QAC

import torch
import torch.nn.functional as F
import numpy as np


class A2C(QAC):
    def _update_actor(self, st0, at0):
        with torch.no_grad():
            q_all = self.critic(st0)
            qsa = q_all.gather(1, at0)
            adv = qsa - q_all.mean(dim=1, keepdim=True)

        lnpi = torch.log(self.actor(st0)).gather(1, at0)
        loss = -(lnpi * adv).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()


if __name__ == "__main__":
    from tqdm import tqdm

    n_envs = 4096
    n_steps = 10000  # 向量 step 数；总环境步数 = n_steps * n_envs

    hidden_dim = 128
    gamma = 0.99
    actor_lr = 3e-4
    critic_lr = 1e-2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a2c = A2C(n_envs, hidden_dim, gamma, actor_lr, critic_lr, device=device)
    a2c.reset(seed=0)

    pbar = tqdm(range(1, n_steps + 1), desc="训练", unit="步")
    for it in pbar:
        a2c.train()
        if it % 500 == 0:
            scorces = a2c.evaluate()
            pbar.set_postfix(
                score=f"{np.mean(scorces):.1f}",
                std=f"{np.std(scorces):.1f}",
            )

    scores = a2c.evaluate()
    print(f"贪心评估: {scores.mean():.1f} ± {scores.std():.1f}")

    a2c.play()
