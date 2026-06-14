import torch

def dice_loss_binary(pred, target):
    smooth = 1e-8
    num = pred.size(0)
    m1 = pred.reshape(num, -1)
    m2 = target.reshape(num, -1)
    intersection = m1 * m2

    intersection = torch.sum(intersection, dim=1, keepdim=True)
    m1 = torch.sum(m1, dim=1, keepdim=True)
    m2 = torch.sum(m2, dim=1, keepdim=True)

    loss = (2. * intersection + smooth) / (m1 + m2 + smooth)
    return 1 - torch.mean(loss, dim=0, keepdim=False)

def dice_loss_multiclass(input, target):
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_loss_binary(input[:, channel, ...], target[:, channel, ...])

    return dice / input.shape[1]

def dice_loss_multiclass_time(input, target):
    dice = 0
    for frame in range(input.shape[2]):
        dice += dice_loss_multiclass(input[:, :, frame, ...], target[:, :, frame, ...])

    return dice / input.shape[2]
