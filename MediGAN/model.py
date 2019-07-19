"""
Model for a Generative Adversarial Network trained to generate medical images
and their accompanying segmentation masks
"""
import torch.nn as nn
import torch.nn.functional as F
import image_processing
import torch.utils as utils
from torchvision import transforms
import torch.optim as optim
import torch

class CycleGAN(nn.Module):

    def __init__(self, image_channels, mask_channels, image_size):
        super(CycleGAN, self).__init__()
        self.image_channels = image_channels
        self.mask_channels = mask_channels
        self.image_size = image_size

        # Make Generators
        self.image_generator = Generator(mask_channels, image_channels, image_size)
        # Swaps because mask_generator is generating three channel image from one channel input
        self.mask_generator = Generator(image_channels, mask_channels, image_size)

        # Make Discriminators
        self.image_discriminator(image_channels)
        self.mask_discriminator(mask_channels)

    def forward(self, image_tensor, mask_tensor):
        predicted_image = self.image_generator(mask_tensor)
        predicted_mask = self.mask_generator(image_tensor)
        image_discriminator_prob = self.image_discriminator(image_tensor)
        mask_discriminator_prob = self.mask_discriminator(mask_tensor)
        fake_image_discriminator_prob = self.image_discriminator(predicted_image)
        fake_mask_discriminator_prob = self.mask_discriminator(predicted_mask)
        return image_discriminator_prob, mask_discriminator_prob, fake_image_discriminator_prob, fake_mask_discriminator_prob

    class Generator(nn.Module):

        def __init__(self, input_channels, out_channels, image_size):
            super(Generator, self).__init__()
            self.input_channels = input_channels
            # Total number of pixels in image
            self.image_size = image_size
            self.out_channels = out_channels # Three because this is generating rgb image from mask
            self.conv1 = nn.Conv2d(input_channels, 256, 3)
            # Fully connected layer from convolution feature maps to each pixel in generated image
            self.fc1 = nn.Linear(5, self.image_size * self.out_channels)

        def forward(self, input_tensor):
            out = F.relu(self.conv1(input_tensor))
            out = self.fc1(out)
            return out


    class Discriminator(nn.Module):

        def __init__(self, input_channels):
            super(Discriminator, self).__init__()
            self.input_channels = input_channels
            self.conv1 = nn.Conv2d(input_channels, 512, 3)
            self.fc1 = nn.Linear(5, 2)

        def forward(self, input_tensor):
            out = F.relu(self.conv1(input_tensor))
            out = F.softmax(self.fc1(out))
            return out

def train():
    # TODO: Add in more transforms for data augmentation
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = image_processing.NucleiDataset("../data/Tissue-images", "../data/full_masks")
    nuclei_dataloader = utils.data.DataLoader(dataset, shuffle=True)
    cycle_gan = CycleGAN(3, 1, 262144)
    optimizer = optim.Adam(cycle_gan.parameters(), lr=.001)

    for i, batch in enumerate(nuclei_dataloader):
        image, mask = batch
        im_discrim_prob, mask_discrim_prob, f_im_discrim_prob, f_mask_discrim_prob = cycle_gan(image, mask)
        # Build up losses 
        im_to_mask_loss = torch.log(1-f_im_discrim_prob) + torch.log(im_discrim_prob)
        mask_to_im_loss = torch.log(1-f_mask_discrim_prob) + torch.log(mask_discrim_prob)
        # TODO: Add in both cyclic consistency losses
        loss = im_to_mask_loss + mask_to_im_loss
