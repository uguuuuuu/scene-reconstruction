import os
from .util import round_up
from .color import autoexposure
from .model import UNet
from .color import PUTransferFunction
from . import tza
import torch
import torch.nn.functional as F

def load_denoiser(input_type, device='cuda'):
    if input_type == 'hdr':
        input_channels = 3
    elif input_type == 'hdr_alb' or input_type == 'hdr_nrm':
        input_channels = 6
    elif input_type == 'hdr_alb_nrm':
        input_channels = 9
    else:
        raise NotImplementedError('Unknown model type!')

    dirname = os.path.dirname(__file__)

    model = UNet(input_channels)
    reader = tza.Reader(f'{dirname}/weights/rt_{input_type}.tza')
    weights = {}
    for key in reader._table:
        weights[key] = torch.tensor(reader[key][0])
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    transfer_fn = PUTransferFunction()
    def denoise(input: torch.Tensor):
        assert(input.dim() == 4)
        image = input.clone()
        if input_type == 'hdr_nrm' or input_type == 'hdr_alb_nrm':
            nrm = image[...,-3:]
            nrm = nrm + 1
            nrm = nrm / 2
            image[...,-3:] = nrm

        exposure = autoexposure(image)
        image = image.permute(0,3,1,2) # (N, H, W, C) -> (N, C, H, W)

        color = image[:,:3,...]
        color = color * exposure
        color = transfer_fn.forward(color)
        image[:,:3,...] = color

        shape = image.shape
        image = F.pad(image, (0, round_up(shape[3], model.alignment) - shape[3],
                          0, round_up(shape[2], model.alignment) - shape[2]))

        output = model(image)
        output = output[:, :, :shape[2], :shape[3]]
        output = torch.clamp(output, min=0.)
        output = transfer_fn.inverse(output)
        output /= exposure

        return output.permute(0,2,3,1)

    return denoise