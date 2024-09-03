import torch
import torch.nn.functional as F


def realEPE(output, target):
    return EPE(output, target, mean=True)


def EPE(output, target, mean=True):
    EPE_map1 = torch.norm(target-output, 2, 1)
    batch_size = EPE_map1.size(0)
    if mean:
        return EPE_map1.mean()
    else:
        return EPE_map1.sum()/batch_size


def mapEPE(target_fusion, target_Gmap, output_Gmap, output_fusion, mean=True):
    target_Gmap = torch.round(target_Gmap, decimals=1)

    dep = 5 - 4 * 2 * abs(target_Gmap-0.5)

    EPE_map0 = torch.norm(target_fusion - output_fusion, 2, 1)

    dep = dep.squeeze(1)


    EPE_map1 = (dep[:,:,:,0] / dep.mean() * torch.norm(target_fusion - output_fusion, 2, 1) + dep[:,:,:,1] / dep.mean() * torch.norm(target_fusion - output_fusion, 2, 1) + dep[:,:,:,2] / dep.mean() * torch.norm(target_fusion - output_fusion, 2, 1)) / 3

    m = ((target_Gmap[:,:,:,:,0] - output_Gmap) + (target_Gmap[:,:,:,:,1] - output_Gmap) + (target_Gmap[:,:,:,:,2] - output_Gmap))/3

    EPE_map2 = torch.norm(m, 2, 1)


    batch_size = EPE_map1.size(0)
    if mean:
        return 0.2 * EPE_map0.mean() + 0.2 * EPE_map1.mean() + 0.2 * EPE_map2.mean()
    else:
        return (0.2 * EPE_map0.sum() + 0.2 * EPE_map1.sum() + 0.2 * EPE_map2.sum())/batch_size
#
