"""Loss functions."""

import torch
from iresnet import iresnet100
import torch.nn.functional as F


class IDLoss(torch.nn.Module):
    """Face ID loss using Glint360K pretrained r100 model."""

    def __init__(self):
        super(IDLoss, self).__init__()
        self.facenet = iresnet100().cuda()
        self.facenet.load_state_dict(torch.load("./glint360k_cosface_r100_fp16_0.1.pth"))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112)).cuda()
        self.facenet.eval()

    def extract_feats(self, x):
        """Extract features. Expect input in square shape."""
        if x.shape[3] != 256:
            x = F.interpolate(x, (256, 256), mode="bilinear", align_corners=True)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats # (1, 512)

    def forward(self, y_hat, y):
        """Calculate the ID loss.
        y_hat: The generated image. Assuming in [-1, 1].
        y: the target image. Assuming in [-1, 1].
        Returns:
            The loss: 1 - dot product
        """
        with torch.no_grad():
            y_feats = self.extract_feats(y) # (N, 512)
        y_hat_feats = self.extract_feats(y_hat)
        loss = (1 - (y_hat_feats * y_feats).sum(1)).mean()
        return loss

