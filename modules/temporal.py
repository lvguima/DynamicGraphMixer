import torch.nn as nn

from layers.TemporalEncoder import TemporalEncoderTCN, TemporalEncoderLinear


class TemporalEncoderWrapper(nn.Module):
    def __init__(self, configs):
        super().__init__()
        temporal_encoder = getattr(configs, "temporal_encoder", "tcn").lower()
        if temporal_encoder == "tcn":
            tcn_kernel = int(getattr(configs, "tcn_kernel", 3))
            tcn_dilation = int(getattr(configs, "tcn_dilation", 2))
            self.encoder = TemporalEncoderTCN(
                d_model=configs.d_model,
                num_layers=configs.e_layers,
                kernel_size=tcn_kernel,
                dilation_base=tcn_dilation,
                dropout=configs.dropout,
            )
        elif temporal_encoder == "linear":
            self.encoder = TemporalEncoderLinear(d_model=configs.d_model)
        else:
            raise ValueError("temporal_encoder must be 'tcn' or 'linear'.")

    def forward(self, x):
        return self.encoder(x)
