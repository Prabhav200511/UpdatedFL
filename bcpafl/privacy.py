# privacy.py - Renyi DP accountant for optional DP-SGD (reused from ProxyFL v1)
#
# Tracks cumulative privacy expenditure across training steps using
# the RDP framework (Mironov, 2017) applied to the subsampled Gaussian
# mechanism, then converts to (Îµ, Î´)-DP via the optimal conversion
# of Balle et al. (2020).

import math


class RDPAccountant:
    """RÃ©nyi Differential Privacy accountant for the subsampled Gaussian mechanism.

    Tracks cumulative RDP across DP-SGD training steps and converts to
    (Îµ, Î´)-DP.  Privacy cost increases monotonically with each step.

    Args:
        noise_multiplier: Gaussian noise scale Ïƒ relative to the clip norm.
        sample_rate:      Poisson sub-sampling rate q = batch_size / dataset_size.
        delta:            Target Î´ for (Îµ, Î´)-DP conversion.
    """

    def __init__(self, noise_multiplier, sample_rate, delta=1e-5):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.steps = 0
        # Evaluate RDP at a range of integer orders and keep the tightest Îµ.
        self.orders = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20,
                       24, 28, 32, 48, 64, 128, 256]

    def step(self, num_steps=1):
        """Record that *num_steps* noisy gradient updates were performed."""
        self.steps += num_steps

    def get_epsilon(self):
        """Return current cumulative Îµ at the configured Î´.

        Returns:
            float: The privacy parameter Îµ (lower is more private).
                   Returns inf when noise_multiplier is 0.
        """
        if self.noise_multiplier == 0:
            return float('inf')
        if self.steps == 0:
            return 0.0
        rdp_values = self._compute_rdp()
        return self._rdp_to_epsilon(rdp_values)

    def get_privacy_spent(self):
        """Return (epsilon, delta) tuple."""
        return self.get_epsilon(), self.delta

    # ------------------------------------------------------------------
    # Internal: RDP computation for the subsampled Gaussian mechanism
    # ------------------------------------------------------------------
    def _compute_rdp(self):
        """Compute cumulative RDP at each order (composition = sum over steps)."""
        return [self._rdp_one_order(alpha) * self.steps
                for alpha in self.orders]

    def _rdp_one_order(self, alpha):
        """RDP at integer order Î± for one step of the subsampled Gaussian mechanism.

        Uses the analytical formula from Mironov (2017), Proposition 3.
        For sub-sampling rate q and noise multiplier Ïƒ:

            RDP_Î± = (1/(Î±-1)) Â· log Î£_{j=0}^{Î±} C(Î±,j) Â· q^j Â· (1-q)^{Î±-j}
                                                  Â· exp(j(j-1) / (2ÏƒÂ²))
        """
        q = self.sample_rate
        sigma = self.noise_multiplier

        if q == 0:
            return 0.0
        if sigma == 0:
            return float('inf')

        # Non-subsampled (q=1): RDP_Î± = Î± / (2ÏƒÂ²)
        if q >= 1.0:
            return alpha / (2.0 * sigma ** 2)

        alpha_int = int(alpha)
        log_terms = []
        for j in range(alpha_int + 1):
            # log C(Î±, j) via log-gamma
            log_comb = (math.lgamma(alpha_int + 1)
                        - math.lgamma(j + 1)
                        - math.lgamma(alpha_int - j + 1))
            # j Â· log(q)
            log_q = j * math.log(q) if j > 0 else 0.0
            # (Î± - j) Â· log(1 - q)
            log_1mq = ((alpha_int - j) * math.log(1.0 - q)
                       if (alpha_int - j) > 0 else 0.0)
            # j(j-1) / (2ÏƒÂ²)
            log_exp = j * (j - 1) / (2.0 * sigma ** 2)

            log_terms.append(log_comb + log_q + log_1mq + log_exp)

        # Numerically stable log-sum-exp
        max_log = max(log_terms)
        lse = max_log + math.log(
            sum(math.exp(t - max_log) for t in log_terms))

        return max(0.0, lse / (alpha_int - 1))

    def _rdp_to_epsilon(self, rdp_values):
        """Convert RDP guarantees to (Îµ, Î´)-DP, returning the tightest Îµ.

        Uses the optimal conversion from Balle et al. (2020), Proposition 3:
            Îµ = RDP(Î±) âˆ’ log(Î´) / (Î± âˆ’ 1) + log(1 âˆ’ 1/Î±)
        """
        eps_candidates = []
        for alpha, rdp in zip(self.orders, rdp_values):
            if rdp == float('inf'):
                eps_candidates.append(float('inf'))
                continue
            eps = (rdp
                   - math.log(self.delta) / (alpha - 1)
                   + math.log(1.0 - 1.0 / alpha))
            eps_candidates.append(max(0.0, eps))
        return min(eps_candidates)


def epsilon_after(steps, noise_multiplier, sample_rate, delta):
    """Cumulative epsilon of ``steps`` subsampled-Gaussian DP-SGD steps."""
    accountant = RDPAccountant(noise_multiplier, sample_rate, delta)
    accountant.step(steps)
    return accountant.get_epsilon()