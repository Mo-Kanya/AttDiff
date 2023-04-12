# Classifier
from mobilenet_classifier import MobileNet

from torch.optim import Adam
from utils.data_loader import get_dataloader

import hydra

def train(epochs, loader, model, optimizer, classifier, rec_scaling, kl_scaling):
    for epoch in range(epochs):
        for batch in loader:
            # Preprocess and obtain real images
            real_images = batch.cuda()

            # TODO - not sure if this is necessary
            real_images.requires_grad_()

            # Encode real images using diffusion's encoder
            encoder_output = model.cond_stage_model.encode(real_images)
            real_classified_logits = classifier.classify_images(real_images)
            style = torch.cat((encoder_output, real_classified_logits), dim=1)

            # Generate images using the modified latent vectors
            generated_images = model.generate(style)

            # Calculate StylEx-related losses
            rec_loss = rec_scaling * reconstruction_loss(real_images, generated_images, encoder_output, model.encoder(generated_images))
            kl_loss = kl_scaling * classifier_kl_loss(real_classified_logits, classifier.classify_images(generated_images))

            # Calculate Latent Diffusion Model loss
            diffusion_loss = latent_diffusion_loss(real_images, generated_images)

            # Combine losses and backpropagate
            total_loss = rec_loss + kl_loss + diffusion_loss
            total_loss.backward()

            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()


        # Optional: Evaluate and/or save model periodically
        # evaluate_and_save_model(model, epoch)

@hydra.main(config_path='configs', config_name='diffusion_ex')
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    # Dataloader
    loader = get_dataloader(cfg)

    # Model
    model = get_model(cfg)

    # Classifier
    classifier = MobileNet(cfg.classifier_path, output_size=cfg.num_classes, image_size=cfg.image_size)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.9)) 

    # Train
    train(cfg.epochs, loader, model, optimizer, classifier, cfg.rec_scaling, cfg.kl_scaling)

