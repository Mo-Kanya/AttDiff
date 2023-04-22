import lpips
import torch
from torch import nn
import torch.nn.functional as F


# helper functions
def lpips_normalize(images):
    """
    Function that scales images to be in range [-1, 1] (needed for LPIPS alex model)
    """
    images_flattened = images.view(images.shape[0], -1)
    _max = images_flattened.max(dim=1)[0].view(-1, 1, 1, 1)
    _min = images_flattened.min(dim=1)[0].view(-1, 1, 1, 1)
    return (images - _min) / (_max - _min) * 2 - 1


lpips_loss = lpips.LPIPS(net="alex").cuda(0)  # image should be RGB, IMPORTANT: normalized to [-1,1]
l1_loss = nn.L1Loss()

# loss functions
def reconstruction_loss(
    real_images: torch.Tensor, 
    generated_images: torch.Tensor, 
    encoder_w: torch.Tensor,
    generated_images_w: torch.Tensor
    ):
    """
    Input: 
        1. Real image batch
        2. Generated image batch
        3. Encoder output for real image batch passed through 
        4. Encoder output for generated image batch passed through
    Output: Reconstruction Loss
    """

    real_images_norm = lpips_normalize(real_images.contiguous())
    generated_images_norm = lpips_normalize(generated_images.contiguous())

    # LPIPS reconstruction loss
    loss = 0.1 * lpips_loss(real_images_norm, generated_images_norm).mean() 
    loss += 0.1 * l1_loss(encoder_w, generated_images_w)
    loss += 1 * l1_loss(real_images, generated_images)
    return loss


kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
def classifier_kl_loss(real_classifier_logits, fake_classifier_logits):
    """
    Input:
        1. Real image batch logits
        2. Generated image batch logits
    Output: KL loss
    """
    # Convert logits to log_softmax and then KL loss

    real_classifier_probabilities = F.log_softmax(real_classifier_logits, dim=1)
    fake_classifier_probabilities = F.log_softmax(fake_classifier_logits, dim=1)

    loss = kl_loss(fake_classifier_probabilities, real_classifier_probabilities)
    return loss

