import torch
import torch.nn as nn
import threading
from .resnet import resnet
from .model import Net, apply_attention
from .utils import get_transform

import inspect
from typing import Any


class InputIdentity(nn.Module):
    def __init__(self, input_name: str) -> None:
        super().__init__()
        self.input_name = input_name

    def forward(self, x):
        return x


class ModelInputWrapper(nn.Module):
    def __init__(self, module_to_wrap: nn.Module) -> None:
        super().__init__()
        self.module = module_to_wrap

        self.arg_name_list = inspect.getfullargspec(
            module_to_wrap.forward).args[1:]
        self.input_maps = nn.ModuleDict(
            {arg_name: InputIdentity(arg_name)
             for arg_name in self.arg_name_list}
        )

    def forward(self, *args, **kwargs) -> Any:
        args = list(args)
        for idx, (arg_name, arg) in enumerate(zip(self.arg_name_list, args)):
            args[idx] = self.input_maps[arg_name](arg)

        for arg_name in kwargs.keys():
            kwargs[arg_name] = self.input_maps[arg_name](kwargs[arg_name])

        return self.module(*tuple(args), **kwargs)


class ResNetLayer4(nn.Module):
    def __init__(self, device):
        super(ResNetLayer4, self).__init__()

        self.r_model = resnet.resnet152(pretrained=True)
        self.r_model.eval()
        self.r_model.to(device)

        self.buffer = {}
        lock = threading.Lock()

        def save_output(module, input, output):
            with lock:
                self.buffer[output.device] = output

        self.r_model.layer4.register_forward_hook(save_output)

    def forward(self, X):
        self.r_model(X)
        return self.buffer[X.device]


class VisualResnetForQuestionAnswering(Net):
    def __init__(self, processor, device):
        num_tokens = processor.num_tokens
        vqa_net = torch.nn.DataParallel(Net(num_tokens))
        vqa_net.load_state_dict(processor.saved_state['weights'])
        vqa_net.to(device)
        vqa_net.eval()
        embedding_tokens = vqa_net.module.text.embedding.num_embeddings

        super().__init__(embedding_tokens)
        self.vqa_net = vqa_net
        self.resnet_layer4 = ResNetLayer4(device)
        self.wrapped = False

    def forward(self, v, q, q_len):

        assert self.wrapped, "This instance must be wrapped before being called. Call wrap_model() on this instance after creation"

        q = self.text(q, list(q_len.data))
        v = self.resnet_layer4(v)

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        a = self.attention(v, q)
        v = apply_attention(v, a)
        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer


class VQAProcessor(nn.Module):
    def __init__(self, state_dict, device):
        self.device = device
        self.saved_state = torch.load(state_dict)
        self.vocab = self.saved_state['vocab']
        self.token_to_index = self.vocab['question']
        self.answer_to_index = self.vocab['answer']
        self.num_tokens = len(self.token_to_index) + 1
        self.answer_words = ['unk'] * len(self.answer_to_index)
        for w, idx in self.answer_to_index.items():
            self.answer_words[idx] = w

    def encode_question(self, question):
        question_arr = question.lower().split()
        vec = torch.zeros(len(question_arr)).long()
        for i, token in enumerate(question_arr):
            index = self.token_to_index.get(token, 0)
            vec[i] = index
        return vec, torch.tensor(len(question_arr))

    def encode_image(self, img):
        image_size = 448
        central_fraction = 1.0

        transform = get_transform(
            image_size, central_fraction=central_fraction)
        img_transformed = transform(img)
        img_batch = img_transformed.unsqueeze(0).to(self.device)
        return img_batch

    def encode_input(self, img, text):
        q, q_len = self.encode_question(text)
        q = q.to(self.device)
        q_len = q_len.to(self.device)
        image_features = self.encode_image(
            img).requires_grad_().to(self.device)
        return (image_features, q.unsqueeze(0), q_len.unsqueeze(0))


def wrap_model(model: VisualResnetForQuestionAnswering):
    partial_dict = model.vqa_net.state_dict()

    model.wrapped = True
    model = ModelInputWrapper(model)
    model = torch.nn.DataParallel(model)

    state = model.module.state_dict()
    state.update(partial_dict)
    model.module.load_state_dict(state)

    return model
