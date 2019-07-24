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
from PIL import Image

class Generator(nn.Module):

    def __init__(self, in_c, out_c):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 40, 5)
        self.conv1_bn = nn.InstanceNorm2d(40)
        self.conv2 = nn.Conv2d(40, 30, 5)
        self.conv2_bn = nn.InstanceNorm2d(30)
        self.conv3 = nn.Conv2d(30, 20, 4)
        self.conv3_bn = nn.InstanceNorm2d(20)
        self.up1 = nn.ConvTranspose2d(20, 30, 5)
        self.up1_bn = nn.InstanceNorm2d(30)
        self.up2 = nn.ConvTranspose2d(30, 40, 5)
        self.up2_bn = nn.InstanceNorm2d(40)
        self.up3 = nn.ConvTranspose2d(40, out_c, 4)


    def forward(self, input_tensor):
        out = F.relu(self.conv1(input_tensor))
        out = self.conv1_bn(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.relu(self.up1(out))
        out = self.up1_bn(out)
        out = F.relu(self.up2(out))
        out = self.up2_bn(out)
        out = self.up3(out)
        return out

class Discriminator(nn.Module):

    def __init__(self, in_c):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 40, 5)
        self.conv1_bn = nn.InstanceNorm2d(40)
        self.conv2 = nn.Conv2d(40, 30, 5)
        self.conv2_bn = nn.InstanceNorm2d(30)
        self.conv3 = nn.Conv2d(30, 20, 5)
        self.conv3_bn = nn.InstanceNorm2d(20)
        self.fc1 = nn.Linear(5000000,2)

    def forward(self, input_tensor):
        out = F.relu(self.conv1(input_tensor))
        out = self.conv1_bn(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.softmax(self.fc1(out.view(-1)), dim=0)
        return out

def train():

    # Defaults to using gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = 200
    # TODO: Add in more transforms for data augmentation

    # Creates datast and dataloader
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    dataset = image_processing.NucleiDataset("../data/Tissue-images", "../data/full_masks", transform)
    nuclei_dataloader = utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Creates generators and discriminators
    image_gen = Generator(1, 3)
    mask_gen = Generator(3, 1)
    image_disc = Discriminator(3)
    mask_disc = Discriminator(1)

    # Add networks onto gpu
    image_gen.to(device)
    mask_gen.to(device)
    image_disc.to(device)
    mask_disc.to(device)

    cyclic_loss = nn.L1Loss()
    optimizer = optim.Adam(list(image_gen.parameters()) + list(mask_gen.parameters()) + list(image_disc.parameters()) + list(mask_disc.parameters()), lr=.00001)

    for epoch in range(num_epochs):
        for i, batch in enumerate(nuclei_dataloader):
            # Puts inputs onto gpu
            image, mask = batch['image'].to(device), batch['label'].to(device)

            # Make predictions
            predicted_image = image_gen(mask)
            predicted_mask = mask_gen(image)
            im_discrim_prob = image_disc(image)
            mask_discrim_prob = mask_disc(mask)
            f_im_discrim_prob = image_disc(predicted_image)
            f_mask_discrim_prob = mask_disc(predicted_mask)
            recov_image = image_gen(predicted_mask)
            recov_mask = mask_gen(predicted_image)
            print(f_im_discrim_prob)
            print(f_mask_discrim_prob)
            print(im_discrim_prob)
            print(mask_discrim_prob)
            # Build up losses
            im_to_mask_loss = torch.mean(torch.log(1-f_im_discrim_prob[0]) + torch.log(im_discrim_prob[0]))
            mask_to_im_loss = torch.mean(torch.log(1-f_mask_discrim_prob[0]) + torch.log(mask_discrim_prob[0]))
            cyclic_loss_im_to_mask = cyclic_loss(recov_image, image)
            cyclic_loss_mask_to_im = cyclic_loss(recov_mask, mask)
            loss = im_to_mask_loss + mask_to_im_loss + 10*(cyclic_loss_im_to_mask + cyclic_loss_mask_to_im)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 5 == 0:
                image = transforms.ToPILImage()(torch.squeeze(predicted_image.cpu().detach()))
                image.show()

if __name__ == "__main__":
    train()
