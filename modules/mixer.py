import torch
import torch.nn as nn


class GraphMixer(nn.Module):
    def __init__(
        self,
        dropout=0.0,
        gate_mode="none",
        num_vars=None,
        num_tokens=None,
        gate_init=0.0,
        conf_map="pow",
        conf_gamma=1.0,
        conf_w_init=4.0,
        conf_b_init=-4.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate_mode = str(gate_mode).lower()
        self.gate = None
        self.conf_map = str(conf_map).lower()
        self.conf_gamma = float(conf_gamma)
        self.conf_w = None
        self.conf_b = None
        self.last_gate = None
        if self.gate_mode == "none":
            self.gate = None
        elif self.gate_mode == "scalar":
            self.gate = nn.Parameter(torch.tensor(gate_init))
        elif self.gate_mode == "per_var":
            if num_vars is None:
                raise ValueError("num_vars is required for per_var gate_mode")
            self.gate = nn.Parameter(torch.full((1, int(num_vars), 1, 1), gate_init))
        elif self.gate_mode == "per_token":
            if num_vars is None or num_tokens is None:
                raise ValueError("num_vars and num_tokens are required for per_token gate_mode")
            self.gate = nn.Parameter(torch.full((1, int(num_vars), int(num_tokens), 1), gate_init))
        elif self.gate_mode in ("conf_scalar", "conf_per_var"):
            if self.conf_map not in ("pow", "affine"):
                raise ValueError(f"Unsupported gate_conf_map: {self.conf_map}")
            if self.gate_mode == "conf_per_var" and num_vars is None:
                raise ValueError("num_vars is required for conf_per_var gate_mode")
            if self.conf_map == "affine":
                if self.gate_mode == "conf_scalar":
                    self.conf_w = nn.Parameter(torch.tensor(conf_w_init))
                    self.conf_b = nn.Parameter(torch.tensor(conf_b_init))
                else:
                    shape = (1, int(num_vars), 1, 1)
                    self.conf_w = nn.Parameter(torch.full(shape, conf_w_init))
                    self.conf_b = nn.Parameter(torch.full(shape, conf_b_init))
        else:
            raise ValueError(f"Unsupported gate_mode: {self.gate_mode}")

    def _get_gate(self, x, token_offset=0):
        if self.gate is None:
            return None
        gate = torch.sigmoid(self.gate)
        if self.gate_mode == "per_token":
            start = int(token_offset) if token_offset is not None else 0
            end = start + x.shape[2]
            gate = gate[:, :, start:end, :]
        return gate

    def _conf_gate(self, conf):
        if conf is None:
            raise ValueError("conf gate requires confidence input")
        if conf.dim() == 1:
            conf = conf.unsqueeze(-1)
        if self.gate_mode == "conf_scalar":
            if conf.dim() == 2:
                conf = conf.mean(dim=1, keepdim=True)
            gate = conf
        else:
            gate = conf
            if gate.dim() == 2:
                gate = gate.unsqueeze(-1).unsqueeze(-1)
        if self.conf_map == "pow":
            gate = gate.clamp(min=0.0) ** self.conf_gamma
        else:
            if self.conf_w is None or self.conf_b is None:
                raise ValueError("affine conf gate requires conf_w/conf_b parameters")
            gate = torch.sigmoid(gate * self.conf_w + self.conf_b)
        if self.gate_mode == "conf_scalar":
            gate = gate.view(gate.shape[0], 1, 1, 1)
        return gate

    def forward(self, adj, x, token_offset=0, conf=None, force_gate_zero=False):
        # adj: [B, C, C], x: [B, C, T, D]
        mixed = torch.einsum("bij,bjtd->bitd", adj, x)
        gate = None
        if force_gate_zero:
            gate = torch.zeros(
                (x.shape[0], 1, 1, 1),
                dtype=x.dtype,
                device=x.device,
            )
        elif self.gate_mode in ("conf_scalar", "conf_per_var"):
            gate = self._conf_gate(conf)
        else:
            gate = self._get_gate(x, token_offset=token_offset)
        if gate is not None:
            mixed = mixed * gate
        mixed = self.dropout(mixed)
        self.last_gate = gate.detach() if gate is not None else None
        return x + mixed
