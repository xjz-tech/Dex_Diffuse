import torch
import torch.nn as nn
import torchvision

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


class DinoV2Encoder(nn.Module):
    """DINOv2 image encoder with an optionally frozen backbone."""

    def __init__(self, model: nn.Module, freeze: bool = True):
        super().__init__()
        self.model = model
        self.freeze = freeze
        if freeze:
            self.model.eval()
            self.model.requires_grad_(False)

    def _forward_impl(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            for key in ("x_norm_clstoken", "cls_token", "last_hidden_state"):
                if key in output:
                    output = output[key]
                    break
            else:
                raise RuntimeError(
                    f"Unsupported DINOv2 output keys: {list(output.keys())}"
                )
        elif isinstance(output, (tuple, list)):
            output = output[0]

        if output.ndim == 3:
            output = output[:, 0]
        elif output.ndim > 2:
            output = torch.flatten(output, start_dim=1)
        return output

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                return self._forward_impl(x).detach()
        return self._forward_impl(x)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self


def get_dinov2(
        name="dinov2_vits14",
        pretrained=True,
        freeze=True,
        repo_or_dir="facebookresearch/dinov2",
        source="github",
        weights=None,
        trust_repo=True,
        skip_validation=True,
        force_reload=False,
        **kwargs):
    """Load a DINOv2 backbone through torch.hub."""
    if weights is not None:
        kwargs["weights"] = weights
    model = torch.hub.load(
        repo_or_dir,
        name,
        pretrained=pretrained,
        source=source,
        trust_repo=trust_repo,
        skip_validation=skip_validation,
        force_reload=force_reload,
        **kwargs,
    )
    return DinoV2Encoder(model=model, freeze=freeze)

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model
