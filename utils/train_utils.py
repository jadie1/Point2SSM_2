# Source: https://github.com/paul007pl/VRCNet/blob/main/utils/train_utils.py
import torch

class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_model(path, net, net_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.state_dict(),
                    'D_state_dict': net_d.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.state_dict()}, path)

def get_random_rot(self, deg=15):
    deg = torch.deg2rad(torch.tensor(deg))
    theta_x = torch.distributions.Uniform(low=-1*deg, high=deg).sample()
    theta_y = torch.distributions.Uniform(low=-1*deg, high=deg).sample()
    theta_z = torch.distributions.Uniform(low=-1*deg, high=deg).sample()
    R1 = torch.eye(3)
    R1[1, 1] = torch.cos(theta_x)
    R1[2, 2] = torch.cos(theta_x)
    R1[1, 2] = -1*torch.sin(theta_x)
    R1[2, 1] = torch.sin(theta_x)
    R2 = torch.eye(3)
    R2[0, 0] = torch.cos(theta_y)
    R2[2, 2] = torch.cos(theta_y)
    R2[2, 0] = -1*torch.sin(theta_y)
    R2[0, 2] = torch.sin(theta_y)
    R3 = torch.eye(3)
    R3[1, 1] = torch.cos(theta_z)
    R3[2, 2] = torch.cos(theta_z)
    R3[1, 2] = -1*torch.sin(theta_z)
    R3[2, 1] = torch.sin(theta_z)
    R =torch.matmul(torch.matmul(R1, R2), R3)
    inv_R = torch.linalg.inv(R)
    return R, inv_R