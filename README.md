# ProxyFL Version 2 — BC-PAFL

**Blockchain-assisted, POMDP-driven Adaptive Federated Learning for VANETs.**

ProxyFL v2 is a from-scratch implementation of the BC-PAFL architecture, built on the parts of
[ProxyFL](https://github.com/Prabhav200511/ProxyFL) that match it. RSUs no longer take every
vehicle that happens to be in range. Each RSU runs a **POMDP controller** that keeps a belief
over each vehicle's hidden state — trust, data utility, model uncertainty and remaining
connection time. From that belief it predicts whether the vehicle will stay connected long
enough to deliver its update, then selects vehicles, compression, aggregation strategy and
local-update frequency to maximise a discounted reward. Before any of this, a vehicle must
authenticate with a **random, blockchain-registered pseudonym** that RSUs check for
authenticity, freshness and revocation without learning the vehicle's real identity.

```text
ProxyFL/            <- original project (unchanged)
ProxyFL-Version2/   <- this repository
```

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install -r requirements.txt
python main.py                          # BC-PAFL, 10 rounds, 30 vehicles, 5 RSUs
python main.py --compare --rounds 12    # BC-PAFL vs. random / all / greedy selection
python -m pytest                        # 38 tests
```

Useful flags: `--rounds`, `--vehicles`, `--seed`, `--selection {pomdp,random,all,greedy}`,
`--malicious 0.1` (fraction of poisoning vehicles), `--alpha 0.5` (Dirichlet label skew; the
default is IID, as in v1), `--lr-decay 0.95` (per-round learning-rate decay, v1's value; 1.0 =
constant), `--no-private` (disable ProxyFL private models), `--dp-noise 1.0` (DP-SGD on the
shared model), `--quiet`, `--no-plots`. Every other parameter is in
[`bcpafl/config.py`](bcpafl/config.py).

Outputs go to `results/<selection>/` (git-ignored): `rounds.csv` (one row per round), `rsu_rounds.csv` (one
row per RSU per round: action, reward terms, selection, dropouts by cause, ξ…),
`summary.json`, and `bcpafl_dashboard.png`. `--compare` also writes `comparison.json` and
`comparison.png`. The v1-style VANET suite writes `vanet_metrics.csv` (one row per node per
round) and the `vanet_*.png` figures, using v1's definitions:

- **Training loss** is each vehicle's private-model DML loss, (1−α)·CE + α·KL. Every vehicle
  trains its private model every round. Selected vehicles train it jointly with M_i. The
  others run the same DML against their local copy of the shared model, which is never uploaded.
- **Private accuracy** is the private model on the vehicle's local held-out split.
- **Global proxy accuracy** is the aggregated model on the attack benchmark
  (`attack1-5_test.csv`), which is what v1's server reports. The held-out test split is in
  `global_test_accuracy_pct` and in `rounds.csv`.
- The per-vehicle energy, latency and crypto figures average only the vehicles that trained M_i
  for upload that round (`fl_participant = 1`).

## Architecture

```text
            +-----------------------------------------------------------+
            |  Permissioned blockchain (PoA: TA, BS, every RSU validate) |
            |  pseudonym records . revocations . trust feedback . model  |
            |  commitments   -- Merkle roots, hash links, 2/3+1 quorum   |
            +-----------------------------------------------------------+
               ^ TA/KGC issue pseudonyms      ^ RSUs post feedback   ^ BS commits sha256(M)
               |                              |                      |
   +-----------+-----+      I2I (wired)      +---------------------+ |
   | TA + KGC        |                       | Base station        |-+
   | real-ID registry|                       | M = sum N_k M_k / N |  Eq. (19)
   | trace / revoke  |                       +----------+----------+
   +-----------------+                                  ^  CLUSTER_UPDATE (signed + encrypted)
                                                        |
        +------------------+------------------+---------+--------+------------------+
        | RSU_0 (0,0)      | RSU_1 (0,1800)   | RSU_2 (1800,0)    | RSU_3 / RSU_4    |
        | POMDP controller: belief (Eq. 6-7) -> q_avail (Eq. 8) -> Psi (Eq. 10)        |
        | action [v, Omega, D, f] (Eq. 9) -> select (Eq. 11, 17, 18)                   |
        | aggregate M_k += sum eps_i dM_i (Eq. 12-13) -> reward (Eq. 14-15) -> learn   |
        +------------------+------------------+-------------------+------------------+
                 ^   V2I: ROUND_START (nonce + BS-signed M), AUTH_BEACON, TRAIN_CONFIG, LOCAL_UPDATE
                 |
        vehicles: random valid pseudonym . noisy GPS/compute/loss report . local DML training
                  Delta M_i compressed (none | Q8 | Q4 | Top-k, error feedback), signed + encrypted
```

### One round (Algorithm 1 + Fig. 2)

| Step | What happens | Code |
|---|---|---|
| 0 | TA/KGC issue fresh pseudonyms to vehicles whose pool expired; pending revocations; one block | `simulation._maintain_pseudonyms`, `identity.TrustedAuthority.provision` |
| 1 | Each RSU broadcasts `ROUND_START`: fresh nonce + the BS-signed global model **M** | `rsu.start_round`, `base_station.global_model_message` |
| 2 | Vehicles in coverage verify both signatures, load M, pick a **random valid pseudonym** and send an encrypted, signed beacon (noisy position/velocity, data size, compute rate, loss and entropy of M on local data) | `vehicle.receive_round_start`, `vehicle.build_beacon` |
| 3 | RSU authenticates against **its own ledger replica**: registered, fresh, not revoked, key matches, batch-verified signature, current nonce and model version | `rsu.authenticate` |
| 4 | Belief handover over I2I if another RSU tracked this pseudonym | `simulation.run_round`, `rsu.handle_belief_request` |
| 5 | Observations O (Eq. 5) → belief update (Eq. 7) → availability q_avail (Eq. 8) → score Ψ (Eq. 10) | `pomdp/controller.py`, `pomdp/belief.py`, `pomdp/scoring.py` |
| 6 | Policy chooses action L = [Ψ_Th, Ω, ∂, f] (Eq. 9); selection v (Eq. 11) under the budgets (Eqs. 17–18) | `pomdp/policy.py`, `ScoreModel.select` |
| 7 | Encrypted `TRAIN_CONFIG` to each selected pseudonym; vehicle trains f epochs, computes ΔM, compresses with Ω, signs and encrypts the upload | `vehicle.train`, `vehicle.build_upload` |
| 8 | Uploads are processed **in simulated-time order**. A vehicle that has left coverage, lost connectivity, run out of frame retries or missed the deadline drops out | `simulation.run_round` (upload phase) |
| 9 | RSU batch-verifies, screens anomalies, computes ε (Eq. 13), aggregates (Eq. 12), evaluates on its validation set and computes the reward (Eqs. 14–15); ξ and Q-policy learn; trust feedback goes on-chain | `rsu.aggregate` |
| 10 | BS verifies cluster updates, applies Eq. (19), evaluates, commits sha256(M) on-chain | `base_station.aggregate` |
| 11 | Block with feedback and the model commitment; the TA traces anomalous pseudonyms to real identities and revokes repeat offenders | `identity.TrustedAuthority.process_feedback` |

## Paper → code map

| Paper | Implementation |
|---|---|
| Eq. (1) local objective | Cross-entropy on D_i in `Vehicle.train` (class-weighted, since class 3 is 0.5% of rows) |
| Eq. (2) weights w_i = \|D_i\|/Σ\|D_j\| | Aggregation strategy `∂ = "data"` |
| Eq. (3) POMDP 7-tuple | `pomdp/controller.py` docstring maps each element |
| Eq. (4) S = [τ, U, μ, ρ] | `pomdp/belief.COMPONENTS`. Discrete levels: 3 each for τ, U, μ; 5 log-spaced levels for ρ (10–810 s) |
| Eq. (5) partial observation | `controller.Observation`: τ̃ from the ledger, Ũ/μ̃ from noisy self-reports, ρ̃ from a straight-line exit time on noisy GPS |
| Eqs. (6)–(7) belief and Bayes update | `BeliefModel.update`. Transitions T are **learned online** (Dirichlet counts from two-slice posteriors). The joint update is exact because T and Z factorise (test: `test_factorised_filter_equals_bruteforce_joint_eq7`) |
| Eq. (8) q_avail = P(Γ_conn > Γ_FL \| β) | `BeliefModel.availability` + `POMDPController.availability` |
| Eq. (9) action [v, Ω, ∂, f] | `policy.Action`; Ω ∈ {none, Q8, Q4, Top-k} in `compression.py` |
| Eq. (10) Ψ with ξ ≥ 0 | `ScoreModel.score`; ξ learned by `ScoreModel.learn` and projected onto [0, ξ_max] |
| Eq. (11) threshold selection | `ScoreModel.select` (Ψ_Th is part of the action) |
| Eqs. (12)–(13) RSU aggregation | `RSU.aggregate` |
| Eqs. (14)–(15) reward, dropout cost | `policy.RewardTerms`, `policy.dropout_cost` |
| Eq. (16) π* = argmax E[Σγ^(t-1)R] | `policy.QPolicy`: semi-gradient Q-learning over belief features |
| Eqs. (17)–(18) budgets | `comm_budget_bytes`, `comp_budget_seconds`, enforced in selection |
| Eq. (19) base-station aggregation | `BaseStation.aggregate` |
| Fig. 1 blockchain / TA / KGC | `blockchain.py`, `identity.py`, `crypto/certificateless.py` |
| Fig. 2 authentication and dynamic pseudonyms | `identity.VehicleWallet.select`, `RSU.authenticate` |

### Where the PDF was ambiguous — choices made

- **Ω = "0"** is read as *no compression* (float32), not "send nothing".
- **Eq. (13)** is printed with `j ≠ i` in the denominator. Taken literally, the coefficients
  would not sum to one, and Eq. (12) is an incremental update of M_k. The normalisation used
  is over all accepted j.
- **Update frequency f** = number of local epochs before uploading.
- **Γ_FL** = local compute time (from the vehicle's reported compute rate) plus the expected
  upload airtime. q_avail is capped by the round deadline and multiplied by the probability
  that the upload's frames get through, so "successfully uploaded" is modelled literally.
- **Trust τ (the paper's ref. [16])** is a beta reputation fed by on-chain RSU feedback
  (accepted +1, dropout −0.5, anomalous −3). The TA carries each identity's reputation into
  its new pseudonyms, so trust survives pseudonym changes without RSUs being able to link them.
- **Data utility U** is Oort-style statistical utility √|D|·RMS(loss of M on D_i), normalised
  per RSU. **Uncertainty μ** is the normalised predictive entropy of M on local data.
- **Algorithm 1, line 20** places the base-station aggregation after the loop, but Fig. 2 shows
  M being broadcast back every round. Global aggregation therefore runs every round.
- **Policy learning:** the Q-function is additive over the four action dimensions, which keeps
  the 72-action space learnable within a few dozen rounds.

## What was reused from ProxyFL v1 and what changed

| Kept (adapted) | Changed or new |
|---|---|
| Certificateless signatures, batch verification, pairwise ECDH, AES-GCM, MIRACL P-256 | Keys belong to **pseudonyms**, not node names; signature moved **inside** the AEAD (v1 limitation L7); nonce-replay cache (L8) |
| TA/KGC AID construction and identity recovery | Dynamic pseudonym pools, validity windows, on-chain registration, TA tracing and revocation |
| Proxy/private models with Deep Mutual Learning, optional DP-SGD and RDP accounting | Shared model enlarged to 4→64→64→6 so compression matters. DML is optional (`--no-private`). As in v1, every vehicle trains its private model every round with a per-round LR decay of 0.95. Only POMDP-selected vehicles upload M_i |
| IID partitions (v1 default) | Dirichlet label skew is available with `--alpha` |
| L2-deviation trust filter | Generalised to model deltas (norm + cosine against the median update) plus a validation check for the cold start |
| Wire codec, 802.11p-style link budget | Frame errors and ARQ now **affect delivery**; broadcast vs unicast airtime |
| Hierarchical vehicle → RSU → server flow | POMDP selection, adaptive aggregation (Eq. 12–13), Eq. 19 weighting, blockchain |
| — | **Real mobility**: v1 had speed fixed at 0, so no vehicle ever left coverage. V2 uses 10–30 m/s with RSU handover |

The transport is a deterministic discrete-event simulation. Every message really is serialised,
encrypted, signed, sent through the channel model and decoded, but it is delivered in-process
rather than over loopback TCP. This makes runs reproducible per seed and lets simulated time
(mobility, compute, airtime) be separate from host wall-clock.

## Security properties exercised in every run

An adversary tries to join each RSU's candidate pool every round. Every attempt below is
rejected, and the rejections are counted by reason in `rounds.csv`:

| Attempt | Rejected by |
|---|---|
| Self-issued key (unregistered pseudonym) | Ledger: `unregistered` |
| Claims an honest vehicle's valid pseudonym with its own key | Envelope key ≠ ledger key: `key_mismatch` |
| Replays a beacon overheard in an earlier round | AEAD nonce cache: `replay`; or pseudonym `expired` |
| Revoked vehicle keeps trying | Ledger: `revoked` |
| Poisoned (−8×, sign-flipped) model update | Anomaly screen → negative on-chain feedback → TA traces and revokes |
| Tampered block on any replica | `Ledger.verify_chain` |
| Forged or stale cluster update to the BS | Ledger key check + signature + model-version check |

## Results

Reference comparison in [`results/reference_seed42_12rounds/`](results/reference_seed42_12rounds/):
seed 42, 30 vehicles, 5 RSUs, 10% malicious vehicles, 12 rounds. It was produced before
private models trained every round, under the earlier defaults (`--alpha 0.5`, no LR decay),
so current runs will not reproduce these exact numbers. The closest current command is
`python main.py --compare --rounds 12 --alpha 0.5 --lr-decay 1.0`. New runs are written to
`results/<selection>/`, which git ignores.

| Selection | Final accuracy | Final macro-F1 | Dropout rate of selected vehicles | Upload |
|---|---|---|---|---|
| **BC-PAFL (POMDP)** | 81.8% | 0.41 | **22%** | 2.19 MB |
| Random | 82.5% | 0.50 | 52% | 1.46 MB |
| All in range | 82.1% | 0.49 | 55% | 1.64 MB |
| Greedy trust | 81.9% | 0.43 | 41% | 1.99 MB |

What this shows, stated plainly:

- **The POMDP does what the paper claims for participation.** It predicts which vehicles will
  stay connected long enough: the dropout rate of selected vehicles falls from 41–55% to 22%.
  The availability Brier score drops over the rounds, and the learned exponent ξ₁
  (availability) grows fastest.
- **It does not yet win on model quality.** At round 12, macro-F1 is below random and
  all-in-range selection. BC-PAFL keeps choosing reliable vehicles, which carries less label
  diversity than random selection, and the reward's ΔAcc term (plain accuracy, as in Eq. 14)
  favours the majority benign class. Adding macro-F1 to the reward and running longer are the
  obvious next experiments.
- It uploads more bytes because it gets more updates delivered (121 selected versus 82–99).
- No malicious update was aggregated under any strategy, and every adversarial join attempt
  was rejected.

## Limitations

- Single process: validators, RSUs and vehicles are objects, not separate hosts. Byte-identical
  signature and key-reconstruction checks are memoised across co-hosted replicas. This changes
  wall-clock time only, never an accept/reject decision.
- The simulation has no road network: mobility is a correlated random walk inside a bounded
  area.
- With 4 input features, the dataset limits the achievable accuracy. A centrally trained
  shared model reaches about 82% accuracy and 0.61 macro-F1 on the held-out test set.
- The RSU validation guard is an extension of the paper, on by default: it shrinks the Eq. (12)
  step when class-weighted validation loss would rise by more than 2%. Set
  `rsu_validation_guard=False` to apply Eq. (12) verbatim.

## Repository layout

```text
main.py                     CLI (single run or --compare)
bcpafl/
  config.py                 all parameters (SimulationConfig)
  simulation.py             Algorithm 1 orchestration, adversary, metrics
  vehicle.py  rsu.py  base_station.py
  pomdp/belief.py           Eqs. 4-8   (state, belief filter, availability)
  pomdp/scoring.py          Eqs. 10-11, 17-18 (score, selection, learnable xi)
  pomdp/policy.py           Eqs. 9, 14-16 (actions, reward, Q-learning)
  pomdp/controller.py       per-RSU controller tying it together
  blockchain.py             PoA ledger, replicas, Merkle roots, quorum endorsements
  identity.py               TA / MVD, pseudonyms, tracing, revocation, wallets
  crypto/certificateless.py certificateless crypto on vendored MIRACL P-256
  secure.py  network.py  channel.py  wire_codec.py
  compression.py            Omega: none / Q8 / Q4 / Top-k + error feedback
  trust.py  mobility.py  data.py  models.py  privacy.py  plotting.py
data/                       VeReMi-derived dataset (from ProxyFL v1)
third_party/miracl/         MIRACL Core licence and notice (Apache 2.0)
tests/                      unit and end-to-end tests
```
