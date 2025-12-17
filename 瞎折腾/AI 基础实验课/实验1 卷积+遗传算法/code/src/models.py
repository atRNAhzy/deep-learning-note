import torch
import torch.nn as nn
try:
    import config as C  # when running as script inside src/
except Exception:  # pragma: no cover
    from . import config as C  # when running as package (python -m src.main)


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 is_batch_norm=False, pooling_type='max',
                 is_dropout=False, dropout_rate=0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

        if pooling_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pooling_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling_type == 'none':
            self.pool = nn.Identity()
        else:
            raise ValueError("Unsupported pooling type. Use 'max', 'avg', or 'none'.")

        self.is_batch_norm = is_batch_norm
        self.is_dropout = is_dropout
        if self.is_batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)
        if self.is_dropout:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        if self.is_batch_norm:
            x = self.batch_norm(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout(x)
        x = self.pool(x)
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels,
                 num_layers=3, num_channels=(64, 128, 256),
                 kernel_size=3, stride=1, padding=1,
                 is_batch_norm=False, pooling_type='max',
                 is_dropout=False, dropout_rate=0.5):
        super().__init__()
        layers = []
        for i in range(num_layers):
            out_channels = num_channels[i]
            layers.append(
                CNNLayer(
                    in_channels, out_channels,
                    kernel_size, stride, padding,
                    is_batch_norm, pooling_type,
                    is_dropout, dropout_rate,
                )
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


class LinearStack(nn.Module):
    def __init__(self, in_features, out_features,
                 num_layers=2, hidden_size=(128, 64),
                 is_dropout=False, dropout_rate=0.5,
                 is_batch_norm=False):
        super().__init__()
        layers = []
        prev = in_features
        for i in range(num_layers):
            layers.append(nn.Linear(prev, hidden_size[i]))
            layers.append(nn.ReLU())
            if is_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size[i]))
            if is_dropout:
                layers.append(nn.Dropout(dropout_rate))
            prev = hidden_size[i]
        self.linear = nn.Sequential(*layers)
        self.out_layer = nn.Linear(prev, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.out_layer(x)
        return x


class CIFAR10CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=C.NUM_CLASSES,
                 cnn_num_layers=3, cnn_num_channels=(64, 128, 256),
                 cnn_is_batch_norm=False, cnn_pooling_type='max',
                 cnn_is_dropout=True, cnn_dropout_rate=0.3,
                 linear_num_layers=2, linear_hidden_size=(128, 64),
                 linear_is_dropout=False, linear_dropout_rate=0.5,
                 linear_is_batch_norm=False,
                 input_size=C.INPUT_SIZE):
        super().__init__()
        self.cnn_block = CNNBlock(
            in_channels,
            num_layers=cnn_num_layers,
            num_channels=cnn_num_channels,
            is_batch_norm=cnn_is_batch_norm,
            pooling_type=cnn_pooling_type,
            is_dropout=cnn_is_dropout,
            dropout_rate=cnn_dropout_rate,
        )
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            feat = self.cnn_block(dummy)
            n_flat = int(feat.shape[1] * feat.shape[2] * feat.shape[3])
        self.linear_layer = LinearStack(
            in_features=n_flat,
            out_features=num_classes,
            num_layers=linear_num_layers,
            hidden_size=linear_hidden_size,
            is_dropout=linear_is_dropout,
            dropout_rate=linear_dropout_rate,
            is_batch_norm=linear_is_batch_norm,
        )

    def forward(self, x):
        x = self.cnn_block(x)
        x = self.flatten(x)
        x = self.linear_layer(x)
        return x


def create_model_from_cfg(cfg: dict) -> nn.Module:
    cnn_num_channels = tuple(C.FIXED_MODEL["cnn_channels"][: cfg["num_cnn_layers"]])
    linear_hidden_size = tuple(C.FIXED_MODEL["linear_hidden"][: cfg["num_linear_layers"]])
    model = CIFAR10CNN(
        cnn_num_layers=cfg["num_cnn_layers"],
        cnn_num_channels=cnn_num_channels,
        cnn_is_batch_norm=cfg["use_bn_cnn"],
        cnn_pooling_type=cfg["pooling_type"],
        cnn_is_dropout=cfg["use_dropout_cnn"],
        cnn_dropout_rate=cfg["dropout_rate"],
        linear_num_layers=cfg["num_linear_layers"],
        linear_hidden_size=linear_hidden_size,
        linear_is_batch_norm=cfg["use_bn_linear"],
        linear_is_dropout=cfg["use_dropout_linear"],
        linear_dropout_rate=cfg["dropout_rate"],
    )
    return model
