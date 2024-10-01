import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm
from typing import Union, List


class GradCamForVQA(nn.Module):
    def __init__(self, model: nn.Module, layer: nn.Module, processor: nn.Module, device: torch.device) -> None:
        super(GradCamForVQA, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.processor = processor

        # 만약 cudnn 백엔드가 활성화되어 있다면, RNN의 역전파에 문제가 발생할 수 있으므로 이를 비활성화
        if torch.backends.cudnn.enabled:
            print("Cannot peform backprop on RNN if torch.backends.cudnn.enabled")
            print("Disabling...")
        torch.backends.cudnn.enabled = False
        self.layer = layer

        self.gradients = None
        self.tensor_hook = None
        self.activation_maps = None

        def forward_hook(model: nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:  # forward_hook은 순전파시, target_layer의 출력을 캡쳐.
            self.activation_maps = out  # target_layer의 출력을 저장
            self.tensor_hook = self.activation_maps.register_hook(self.backward_hook)  # 저장된 출력 tensor에 backward_hook을 등록, gradient를 캡쳐.

        self.layer_hook = self.layer.register_forward_hook(forward_hook)  # target_layer에 위의 forward_hook을 등록함.
        

    def backward_hook(self, grad: torch.Tensor) -> None:  # 역전파 시, activation_map의 gradient를 저장.
        self.gradients = grad  # 역전파 시, 전달되는 Gradient값.


    def _compute_cam(self, encoding: dict, y: Union[torch.tensor, None]) -> torch.tensor:  
        """
        encoding: 이미지와 질문의 인코딩된 입력
        y: 타겟 클래스 인덱스. 없을 경우 모델의 예측 결과를 사용
        """
        out = self.model(*encoding) # out_shape:  torch.Size([1, 3000])
        _, answer_idx = F.softmax(out, dim=1).data.cpu().max(dim=1) # 모델의 출력을 소프트맥스 함수를 통해 확률로 변환, 가장 높은 확률을 가진 클래스의 인덱스를 answer_idx에 저장

        activation_maps = self.activation_maps.detach().cpu() # 캡처된 activation_map을 CPU로 이동, gradient계산에서 분리
        # activation_maps.shape: torch.Size([1, 2048, 14, 14])

        print(f"Answer: {self.processor.answer_words[answer_idx]}") # 예측된 답변의 인덱스를 실제 단어로 변환하여 출력

        
        if y is None:
            out[:, answer_idx].backward() # 만약 타겟 클래스 y가 주어지지 않았다면, 모델의 예측 클래스에 대해 역전파를 수행
        else:
            assert y.shape == torch.Size([1]) # y가 주어졌다면 assert 문을 통해 y의 크기가 올바른지 확인.
            out[:, y].backward()  # 해당 클래스에 대해 역전파를 수행. 

        scores = torch.mean(self.gradients.detach().cpu(), dim=[0, 2, 3]).detach().cpu() # 캡처된 그래디언트를 채널, 높이, 너비 방향으로 평균내어 가중치(scores)를 계산
        # scores.shape: torch.Size([2048])
        
        for i in range(activation_maps.shape[1]):
            activation_maps[:, i, :, :] *= scores[i] # 각 채널의 활성화 맵에 해당 채널의 가중치를 곱해 가중치가 반영된 activation_map을 얻음

        heatmap = torch.mean(activation_maps, dim=1).squeeze() # 채널 방향으로 평균내어 최종 히트맵을 생성, squeeze로 불필요 차원 제거 => 2D Heat_map
        # heatmap_shape: torch.Size([14, 14])
        heatmap = F.relu(heatmap) # 히트맵의 음수 값을 제거하여 시각화를 개선
        return heatmap / torch.max(heatmap) # 히트맵을 최대 값으로 나누어 0과 1 사이로 정규화

    def forward(self, _image: Image.Image, question: str, label: Union[torch.Tensor, None] = None, superimpose: bool = True) -> Image:
        image = _image.copy()
        encoding = self.processor.encode_input(image, question)
        cam = self._compute_cam(encoding, label)  # cam_shape: torch.Size([14, 14])
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
