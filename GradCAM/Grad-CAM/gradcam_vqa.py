from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np


from torchvision import transforms
from PIL import Image
from typing import Union, List

class GradCamForVQA(nn.Module):
    def __init__(self, model: nn.Module, layer: nn.Module, processor: nn.Module, device: torch.device) -> None:
        super(GradCamForVQA, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        if torch.backends.cudnn.enabled:
            print("Cannot peform backprop on RNN if torch.backends.cudnn.enabled")
            print("Disabling...")
        torch.backends.cudnn.enabled = False
        self.layer = layer

        self.gradients = None
        self.tensor_hook = None
        self.activation_maps = None

        def forward_hook(model: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
            self.activation_maps = out
            self.tensor_hook = self.activation_maps.register_hook(
                self.backward_hook)

        self.layer_hook = self.layer.register_forward_hook(forward_hook)
        self.processor = processor

    def backward_hook(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def _compute_cam(self, encoding: dict, y: Union[torch.tensor, None]) -> torch.tensor:
        out = self.model(*encoding)
        _, answer_idx = F.softmax(out, dim=1).data.cpu().max(dim=1)

        activation_maps = self.activation_maps.detach().cpu()

        print(f"Answer: {self.processor.answer_words[answer_idx]}")
        if y is None:
            out[:, answer_idx].backward()
        else:
            assert y.shape == torch.Size([1])
            out[:, y].backward()

        scores = torch.mean(self.gradients.detach().cpu(),
                            dim=[0, 2, 3]).detach().cpu()

        for i in range(activation_maps.shape[1]):
            activation_maps[:, i, :, :] *= scores[i]

        heatmap = torch.mean(activation_maps, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        return heatmap / torch.max(heatmap)

    def forward(self, _image: Image.Image, question: str, label: Union[torch.Tensor, None] = None, superimpose: bool = True) -> Image:
        image = _image.copy()
        encoding = self.processor.encode_input(image, question)
        cam = self._compute_cam(encoding, label)
        inv_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image.size[::-1])
        ])
        cam = inv_transforms(cam)

        if superimpose:
            cmap = cm.get_cmap('jet', 256)
            cam = np.array(cmap(cam))
            cam = Image.fromarray((cam * 255).astype(np.uint8))
            cam.putalpha(128)
            image.paste(cam, (0, 0), cam)
            return image
        else:
            return cam
