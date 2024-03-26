import os
import torch
import torch.nn.functional


def save_model(net, recorder):
    model_dir = os.path.join(recorder.work_dir, "ckpt")
    os.system(f"mkdir -p {model_dir}")
    epoch = recorder.epoch
    ckpt_name = epoch
    torch.save(
        {
            "net": net.state_dict(),
        },
        os.path.join(model_dir, f"{ckpt_name}.pth"))


def load_network(net, model_dir, strict=False):
    weights = torch.load(model_dir)["net"]
    new_weights = {}
    for k, v in weights.items():
        new_k = k.replace("module.", '') if "module" in k else k
        new_weights[new_k] = v
    net.load_state_dict(new_weights, strict=strict)
