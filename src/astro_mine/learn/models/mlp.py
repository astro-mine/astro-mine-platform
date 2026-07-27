"""Feed-forward policy/value building blocks (RM-P1-LEARN-03; learn.md §3).

The shared MLP trunk plus the two heads the baselines compose over the SwarmEnv's
heterogeneous, capability-keyed spaces:

- :class:`MLP` — a plain multilayer perceptron trunk.
- :class:`DictActorCritic` — an actor-critic whose actor emits one categorical head per
  discrete action selector (``kind``/``mode``) and one squashed-Gaussian head per continuous
  block (``goto``/``hop``), matching the Core tagged-union action; its critic is a scalar
  value head. Because SADF agents are heterogeneous (different obs/action dims), each agent
  owns its own net rather than sharing parameters.
- :class:`AgentQNet` — a per-agent Q-net over the discrete ``kind`` selector, the QMIX
  baseline's value head.

An optional recurrent core (:class:`~astro_mine.learn.models.rnn.GRUCore`) and comms module
(:class:`~astro_mine.learn.models.comms.MessageModule`) plug in for partial observability
and the comms-learning research track.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from astro_mine.learn.models.rnn import GRUCore

__all__ = ["MLP", "ActorOutput", "AgentQNet", "DictActorCritic"]


class MLP(nn.Module):
    """A tanh-activated MLP trunk mapping ``in_dim`` → ``out_dim``."""

    def __init__(self, in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class ActorOutput:
    """The per-head distribution parameters + sampled action of a :class:`DictActorCritic`.

    ``action`` is the decoded action sample (``kind``/``mode`` ints, ``goto``/``hop`` arrays)
    ready for ``decode_action``; ``log_prob``/``entropy`` sum over every head; ``value`` is
    the local critic estimate."""

    __slots__ = ("action", "entropy", "log_prob", "value")

    def __init__(
        self,
        action: dict[str, object],
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.action = action
        self.log_prob = log_prob
        self.entropy = entropy
        self.value = value


class DictActorCritic(nn.Module):
    """A heterogeneous-action actor-critic for one agent.

    ``discrete_heads`` maps each discrete selector to its cardinality and ``box_heads`` each
    continuous block to its width; ``comms_dim`` widens the trunk input by an aggregated
    peer-message vector when the comms-learning track is enabled."""

    def __init__(
        self,
        obs_dim: int,
        discrete_heads: Mapping[str, int],
        box_heads: Mapping[str, int],
        hidden_sizes: Sequence[int],
        *,
        use_rnn: bool = False,
        comms_dim: int = 0,
    ) -> None:
        super().__init__()
        hidden = list(hidden_sizes) or [32]
        feat = hidden[-1]
        self.obs_dim = obs_dim
        self.comms_dim = comms_dim
        self.feat_dim = feat
        self.use_rnn = use_rnn
        # Normalize the heterogeneous trunk input (pose in metres, battery in joules, sensor
        # counts, and any aggregated peer-message vector) to O(1) so the unnormalized SI
        # magnitudes cannot blow up the heads.
        trunk_in = obs_dim + comms_dim
        self.obs_norm = nn.LayerNorm(trunk_in)
        self.rnn: GRUCore | None
        if use_rnn:
            self.rnn = GRUCore(trunk_in, feat)
            self.trunk: nn.Module = nn.Identity()
        else:
            self.rnn = None
            self.trunk = MLP(trunk_in, hidden[:-1], feat)
        self.discrete = nn.ModuleDict({k: nn.Linear(feat, n) for k, n in discrete_heads.items()})
        self.box_mean = nn.ModuleDict({k: nn.Linear(feat, d) for k, d in box_heads.items()})
        self.box_log_std = nn.ParameterDict(
            {k: nn.Parameter(torch.zeros(d)) for k, d in box_heads.items()}
        )
        self.value_head = nn.Linear(feat, 1)
        # Head widths kept as plain ints (an nn.ModuleDict value is typed ``Module``, so the
        # export IO declaration would otherwise need a cast on every lookup).
        self.discrete_dims: dict[str, int] = dict(discrete_heads)
        self.box_dims: dict[str, int] = dict(box_heads)

    # --- trunk ----------------------------------------------------------------------

    def trunk_input(self, obs: torch.Tensor, msg: torch.Tensor | None = None) -> torch.Tensor:
        """Assemble the trunk input: the flat observation, widened by the aggregated
        peer-message context when the comms-learning track is enabled (``comms_dim > 0``).

        An isolated agent — or a host with no peers — passes ``msg=None`` and gets the zero
        message vector, exactly the :class:`~astro_mine.learn.models.comms.MessageModule`
        semantics for an agent nothing reached this tick."""
        if self.comms_dim == 0:
            return obs
        if msg is None:
            msg = torch.zeros(obs.shape[0], self.comms_dim, dtype=obs.dtype)
        return torch.cat([obs, msg], dim=-1)

    def features(
        self, obs: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Trunk features (and next hidden state when recurrent).

        ``obs`` is the **assembled** trunk input (:meth:`trunk_input`), i.e. already widened
        by the peer-message context for a comms-learning policy."""
        obs = self.obs_norm(obs)
        if self.rnn is not None:
            h = self.rnn(obs, hidden)
            return h, h
        return self.trunk(obs), None

    def _distributions(
        self, feat: torch.Tensor
    ) -> tuple[dict[str, torch.distributions.Categorical], dict[str, torch.distributions.Normal]]:
        cats = {
            k: torch.distributions.Categorical(logits=head(feat))
            for k, head in self.discrete.items()
        }
        norms = {
            k: torch.distributions.Normal(
                torch.tanh(self.box_mean[k](feat)), self.box_log_std[k].exp()
            )
            for k in self.box_mean
        }
        return cats, norms

    def act(
        self, obs: torch.Tensor, generator: torch.Generator, hidden: torch.Tensor | None = None
    ) -> tuple[ActorOutput, torch.Tensor | None]:
        """Sample an action (seeded ``generator`` for reproducibility) + its log-prob/value."""
        feat, next_h = self.features(obs, hidden)
        cats, norms = self._distributions(feat)
        action: dict[str, object] = {}
        log_prob = torch.zeros((), dtype=torch.float32)
        entropy = torch.zeros((), dtype=torch.float32)
        for k, cat in cats.items():
            probs = cat.probs.detach()
            idx = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            action[k] = int(idx.item())
            log_prob = log_prob + cat.log_prob(idx)
            entropy = entropy + cat.entropy()
        for k, norm in norms.items():
            noise = torch.randn(norm.mean.shape, generator=generator)
            sample = (norm.mean + norm.stddev * noise).clamp(-1.0, 1.0)
            log_prob = log_prob + norm.log_prob(sample).sum(-1)
            entropy = entropy + norm.entropy().sum(-1)
            action[k] = sample.squeeze(0).detach().numpy().astype("float32")
        value = self.value_head(feat).squeeze(-1)
        return ActorOutput(action, log_prob, entropy, value), next_h

    def act_batch(
        self, obs: torch.Tensor, generator: torch.Generator, hidden: torch.Tensor | None = None
    ) -> tuple[list[dict[str, object]], torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sample one action per row of a ``(batch, trunk_in)`` observation in **one** forward.

        The batched-inference kernel the GPU-vectorized executor drives (RM-P1-LEARN-04): the
        whole env batch crosses the net once instead of ``batch`` sequential forwards. Draws
        from the same seeded ``generator`` in the same order as :meth:`act`, so a batch of one
        is byte-identical to the sequential path. Returns the per-row action samples, the
        summed log-probs ``(batch,)``, the critic values ``(batch,)``, and the next hidden
        state when recurrent."""
        feat, next_h = self.features(obs, hidden)
        cats, norms = self._distributions(feat)
        batch = int(obs.shape[0])
        actions: list[dict[str, object]] = [{} for _ in range(batch)]
        log_prob = torch.zeros(batch, dtype=torch.float32)
        for k, cat in cats.items():
            idx = torch.multinomial(cat.probs.detach(), 1, generator=generator).squeeze(-1)
            log_prob = log_prob + cat.log_prob(idx)
            for row in range(batch):
                actions[row][k] = int(idx[row].item())
        for k, norm in norms.items():
            noise = torch.randn(norm.mean.shape, generator=generator)
            sample = (norm.mean + norm.stddev * noise).clamp(-1.0, 1.0)
            log_prob = log_prob + norm.log_prob(sample).sum(-1)
            block = sample.detach().numpy().astype("float32")
            for row in range(batch):
                actions[row][k] = block[row]
        value = self.value_head(feat).squeeze(-1)
        return actions, log_prob, value, next_h

    def greedy(self, obs: torch.Tensor) -> dict[str, object]:
        """The deterministic (argmax / distribution-mean) action for evaluation & export."""
        with torch.no_grad():
            feat, _ = self.features(obs)
            action: dict[str, object] = {}
            for k, head in self.discrete.items():
                action[k] = int(head(feat).argmax(dim=-1).item())
            for k in self.box_mean:
                mean = torch.tanh(self.box_mean[k](feat)).squeeze(0)
                action[k] = mean.numpy().astype("float32")
        return action

    # --- ONNX export IO (RM-P1-LEARN-05) --------------------------------------------

    def export_input_specs(self) -> list[tuple[str, int]]:
        """The ordered ONNX graph **inputs** — ``(name, width)`` — matching
        :meth:`forward_export`'s positional arguments.

        Always ``obs``; plus ``msg`` (the aggregated peer-message context) for a
        comms-learning policy, and ``hidden_in`` (the GRU state) for a recurrent one. The
        state and message tensors are *explicit* graph inputs rather than hidden internals, so
        the Core :class:`~astro_mine.core.policy.model.IoSignature` declares the full tensor
        contract a host must bind (learn.md §5; surrogate.md §11 "served artifact is ONNX")."""
        specs = [("obs", self.obs_dim)]
        if self.comms_dim:
            specs.append(("msg", self.comms_dim))
        if self.use_rnn:
            specs.append(("hidden_in", self.feat_dim))
        return specs

    def export_output_specs(self) -> list[tuple[str, int]]:
        """The ordered ONNX graph **outputs** — per-discrete-head logits, then per-box-head
        squashed means (sorted keys), then ``hidden_out`` for a recurrent policy."""
        specs = [(k, self.discrete_dims[k]) for k in sorted(self.discrete_dims)]
        specs += [(k, self.box_dims[k]) for k in sorted(self.box_dims)]
        if self.use_rnn:
            specs.append(("hidden_out", self.feat_dim))
        return specs

    def forward_export(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Traceable, tensor-only actor forward for the ONNX PolicyPackage graph (LEARN-05).

        Takes the tensors declared by :meth:`export_input_specs`, positionally: ``obs``, then
        ``msg`` (comms-learning), then ``hidden_in`` (recurrent). Returns per-discrete-head
        **logits**, per-box-head **squashed means**, and — when recurrent — the next hidden
        state, in :meth:`export_output_specs` order. No sampling, ``argmax`` or ``.item()``, so
        every graph output is a float tensor and the ONNX-Runtime equivalence check is a clean
        ``allclose``; argmax over the discrete logits is done host-side (matching
        :class:`~astro_mine.core.policy.OnnxPolicy`). Only the decentralized **actor** is
        exported — the CTDE critic value head stays internal.

        The recurrent graph carries its GRU state as an explicit ``hidden_in``/``hidden_out``
        tensor pair (a *stateless* graph over an explicit state), which is what keeps it
        expressible at the pinned opset with a single dynamic (batch) axis — no ONNX ``Loop``,
        no sequence-length axis, and a host that carries the state across calls."""
        obs = inputs[0]
        index = 1
        msg: torch.Tensor | None = None
        if self.comms_dim:
            msg = inputs[index]
            index += 1
        hidden: torch.Tensor | None = None
        if self.use_rnn:
            hidden = inputs[index]
        feat, next_hidden = self.features(self.trunk_input(obs, msg), hidden)
        outputs = [self.discrete[k](feat) for k in sorted(self.discrete)]
        outputs += [torch.tanh(self.box_mean[k](feat)) for k in sorted(self.box_mean)]
        if next_hidden is not None:
            outputs.append(next_hidden)
        return tuple(outputs)

    def evaluate(
        self, obs: torch.Tensor, actions: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored actions: (log_prob, entropy, value) for the PPO update."""
        feat, _ = self.features(obs)
        cats, norms = self._distributions(feat)
        log_prob = torch.zeros(obs.shape[0], dtype=torch.float32)
        entropy = torch.zeros(obs.shape[0], dtype=torch.float32)
        for k, cat in cats.items():
            log_prob = log_prob + cat.log_prob(actions[k])
            entropy = entropy + cat.entropy()
        for k, norm in norms.items():
            log_prob = log_prob + norm.log_prob(actions[k]).sum(-1)
            entropy = entropy + norm.entropy().sum(-1)
        value = self.value_head(feat).squeeze(-1)
        return log_prob, entropy, value


class AgentQNet(nn.Module):
    """A per-agent Q-net over the discrete ``kind`` selector — the QMIX value head."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.obs_norm = nn.LayerNorm(obs_dim)
        self.net = MLP(obs_dim, hidden_sizes, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        q: torch.Tensor = self.net(self.obs_norm(obs))
        return q

    def export_input_specs(self) -> list[tuple[str, int]]:
        """The ordered ONNX graph inputs — the value-based baseline is stateless and
        comms-blind, so a single ``obs`` tensor."""
        return [("obs", self.obs_dim)]

    def export_output_specs(self) -> list[tuple[str, int]]:
        """The ordered ONNX graph outputs — the per-``kind`` Q-values."""
        return [("kind", self.n_actions)]

    def forward_export(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Traceable actor forward for the ONNX PolicyPackage graph (LEARN-05): the per-``kind``
        Q-values as a single output (argmax host-side, matching :class:`OnnxPolicy`)."""
        return (self.forward(inputs[0]),)
