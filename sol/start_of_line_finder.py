import torch
from torch import nn

# from torchvision.models import vgg
from sol import vgg


class StartOfLineFinder(nn.Module):
    def __init__(self, base_0, base_1):
        super().__init__()
        self.cnn = vgg.vgg11()
        self.base_0 = base_0
        self.base_1 = base_1

    def forward(self, img):
        y = self.cnn(img)  # type: torch.Tensor  # size: (N, 5, H, W)

        # restoring y from the w index and the y offset
        priors_0 = torch.arange(0, y.size(2)).type_as(img.data)[None, :, None]  # size: (1, H, 1)  tiling x by y axis
        priors_0 = (priors_0 + 0.5) * self.base_0
        priors_0 = priors_0.expand(y.size(0), priors_0.size(1), y.size(3))  # size: (N, H, W)
        priors_0 = priors_0[:, None, :, :]  # size: (N, 1, H, W)

        # restoring x from the w index and the x offset
        priors_1 = torch.arange(0, y.size(3)).type_as(img.data)[None, None, :]  # size: (1, 1, W)  tiling y by x axis
        priors_1 = (priors_1 + 0.5) * self.base_1
        priors_1 = priors_1.expand(y.size(0), y.size(2), priors_1.size(2))  # size: (N, H, W)
        priors_1 = priors_1[:, None, :, :]  # size: (N, 1, H, W)

        predictions = torch.cat([
            torch.sigmoid(y[:, 0:1, :, :]),
            y[:, 1:2, :, :] + priors_0,
            y[:, 2:3, :, :] + priors_1,
            y[:, 3:4, :, :],
            y[:, 4:5, :, :]
        ], dim=1)  # (N, 5, H, W)

        predictions = predictions.transpose(1, 3).contiguous()  # (N, W, H, 5)
        predictions = predictions.view(predictions.size(0), -1, 5)  # last dim: prob, x, y, rotation, scale

        return predictions
