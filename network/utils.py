import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from scipy import ndimage


class Extractor_DI(nn.Module):
    def __init__(self, n_channels=64):
        super(Extractor_DI, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(2048, n_channels*2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(n_channels*2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(n_channels*2, n_channels*2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(n_channels*2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(n_channels*2, 2048, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(2048),
            )
    def forward(self, f_all):
        f_di = self.inc(f_all)
        return f_di

class Domain_classifier(nn.Module):
    def __init__(self, n_channels=64):
        super(Domain_classifier, self).__init__()
        # self.conv = nn.Sequential(
            # nn.Conv2d(n_channels, n_channels * 2, kernel_size=7, stride=2, padding=3),
            # nn.BatchNorm2d(n_channels * 2),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=7, stride=2, padding=3),
            # nn.BatchNorm2d(n_channels * 4),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=7, stride=2, padding=3),
            # nn.BatchNorm2d(n_channels * 8),
            # nn.ReLU(inplace=True),
        # )
        self.pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(n_channels * 8, 1)
        self.discriminator = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(2048, n_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(n_channels, n_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(n_channels * 2, n_channels * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(n_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(n_channels * 4, n_channels * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(n_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(n_channels * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x1 = self.conv(x)
        # x2 = self.pool(x1).squeeze(-1).squeeze(-1)
        # prob = self.fc(x2)
        x1 = self.discriminator(x)
        x2 = self.pool(x1).view(-1, 512)
        prob = self.fc(x2)
        return self.sig(prob.view(-1))


from typing import Optional, Any, Tuple
import torch.nn as nn
from torch.autograd import Function
import torch

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
    # def forward(self, ctx, input, coeff = 1.):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
    # def backward(self, ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super(_SimpleSegmentationModel, self).__init__()
        self.feature_extractor = backbone
        self.classifier = classifier

        self.E_DI = Extractor_DI()
        self.grl_di = GradientReverseLayer()
        self.grl_ds = GradientReverseLayer()
        self.invariant_classifier = Domain_classifier()
        self.specific_classifier = Domain_classifier()
        self.avgpool = torch.nn.AvgPool2d(28)

    def forward(self, x, step = 1):
        input_shape = x.shape[-2:]
        features = self.feature_extractor(x)

        # if step == 2:             
            # features['out'] = features['out'].detach()

        f_all = features['out']
        f_di = self.E_DI(f_all)
        f_ds = f_all - f_di

        f_di_pool = self.avgpool(f_di).squeeze().squeeze()
        f_ds_pool = self.avgpool(f_ds).squeeze().squeeze()
        # loss_orthogonal = (f_di_pool.square() * f_ds_pool.square()).mean()
        loss_orthogonal = F.cosine_similarity(f_di_pool, f_ds_pool, dim=1)
        f_di_re = self.grl_di(f_di)
        # f_di_re = f_di
        prob_di = self.invariant_classifier(f_di_re)

        # f_ds = grad_reverse(f_ds, 1.0)
        # f_ds = self.grl_ds(f_ds)
        prob_ds = self.specific_classifier(f_ds)



        features['out'] = f_di
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        # att_cacs_map = features['low_level'].cpu().detach().numpy().astype(np.float)
#         att_cacs_map = f_di.cpu().detach().numpy().astype(np.float)

#         att_cacs_map = np.mean(att_cacs_map, axis=1)
#         att_cacs_map = ndimage.interpolation.zoom(att_cacs_map, [1.0, 224 / att_cacs_map.shape[1],
#                                                               224 / att_cacs_map.shape[2]], order=1)
        return x, loss_orthogonal, prob_di, prob_ds


# class _SimpleSegmentationModel(nn.Module):
    # def __init__(self, backbone, classifier):
        # super(_SimpleSegmentationModel, self).__init__()
        # self.backbone = backbone
        # self.classifier = classifier
        
    # def forward(self, x):
        # input_shape = x.shape[-2:]
        # features = self.backbone(x)
        # x = self.classifier(features)
        # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        # att_cacs_map = features['out'].cpu().detach().numpy().astype(np.float)

        # att_cacs_map = np.mean(att_cacs_map, axis=1)
        # att_cacs_map = ndimage.interpolation.zoom(att_cacs_map, [1.0, 224 / att_cacs_map.shape[1],
                                                              # 224 / att_cacs_map.shape[2]], order=1)
        # return x, None, None, None, att_cacs_map


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
