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
import argparse

class Generator(nn.Module):

    def __init__(self, in_c, out_c):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 120, 3)
        self.conv1_bn = nn.BatchNorm2d(120)
        self.conv2 = nn.Conv2d(120, 80, 3)
        self.conv2_bn = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 60, 4)
        self.conv3_bn = nn.BatchNorm2d(60)
        self.up1 = nn.ConvTranspose2d(60, 80, 4)
        self.up1_bn = nn.BatchNorm2d(80)
        self.up2 = nn.ConvTranspose2d(80, 120, 3)
        self.up2_bn = nn.BatchNorm2d(120)
        self.up3 = nn.ConvTranspose2d(120, out_c, 3)


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
        out = F.tanh(self.up3(out))
        return out

class Discriminator(nn.Module):

    def __init__(self, in_c):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 80, 4)
        self.conv1_bn = nn.BatchNorm2d(80)
        self.conv2 = nn.Conv2d(80, 60, 4)
        self.conv2_bn = nn.BatchNorm2d(60)
        self.conv3 = nn.Conv2d(60, 40, 4)
        self.conv3_bn = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(1190720, 2)

    def forward(self, input_tensor):
        out = F.leaky_relu(self.conv1(input_tensor))
        out = self.conv1_bn(out)
        out = F.leaky_relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = F.leaky_relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.softmax(self.fc1(out.view(out.size(0),-1)), dim=1)
        return out

def set_grad(model, state):
    for param in model.parameters():
        param.requires_grad = state

def train():

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, help="Learning rate to use for training", default=.0002)
    parser.add_argument("--show_images", help="shows generated image every 10 epochs", action="store_true")
    parser.add_argument("--checkpoint_frequency", type=int, help="If given, saves a copy of the weights every x epochs, where x is the integer passed in. Default is no checkpoints saved")
    parser.add_argument("--prev_model", help="if given, will load in previous saved model from a .tar file. Argument should be path to .tar file to load")
    args = parser.parse_args()

    # Defaults to using gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 1000
    batch_size = 5
    # TODO: Add in more transforms for data augmentation

    # Creates datast and dataloader
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    dataset = image_processing.NucleiDataset("../data/Tissue-images", "../data/full_masks", transform)
    nuclei_dataloader = utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
    gen_optimizer = optim.Adam(list(image_gen.parameters()) + list(mask_gen.parameters()), lr=args.learning_rate)
    disc_optimizer = optim.Adam(list(image_disc.parameters()) + list(mask_disc.parameters()), lr=.0001)

    prev_epoch = 0
    # Loads in previous model if given
    if args.prev_model:
        checkpoint = torch.load(args.prev_model)
        image_gen.load_state_dict(checkpoint['image_gen_model'])
        mask_gen.load_state_dict(checkpoint['mask_gen_model'])
        image_disc.load_state_dict(checkpoint['image_disc_model'])
        mask_disc.load_state_dict(checkpoint['mask_disc_model'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_model'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_model'])
        prev_epoch = checkpoint['epoch']
        # TODO: Use loss here after adding in loss graphin

    for epoch in range(prev_epoch, num_epochs):
        for i, batch in enumerate(nuclei_dataloader):
            print("Epoch: ", epoch)
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
            # Disable discriminator gradients for generator update
            set_grad(image_disc, False)
            set_grad(mask_disc, False)
            # Get generator losses
            gen_optimizer.zero_grad()
            im_to_mask_gen_loss = torch.mean(-(f_im_discrim_prob[0])**2 + (im_discrim_prob[0])**2)
            mask_to_im_gen_loss = torch.mean(-(f_mask_discrim_prob[0])**2 + mask_discrim_prob[0]**2)
            # Get cyclic losses
            cyclic_loss_im_to_mask = cyclic_loss(recov_image, image)
            cyclic_loss_mask_to_im = cyclic_loss(recov_mask, mask)
            # Total up gen losses and optimize
            gen_loss = im_to_mask_gen_loss + mask_to_im_gen_loss + 10*(cyclic_loss_im_to_mask + cyclic_loss_mask_to_im)
            gen_loss.backward(retain_graph=True)
            gen_optimizer.step()

            # Turn gradients back on for disciminators
            set_grad(image_disc, True)
            set_grad(mask_disc, True)

            # Get discriminator losses
            disc_optimizer.zero_grad()
            im_discrim_loss = torch.mean((1-im_discrim_prob[0])**2 + (f_im_discrim_prob[0])**2)
            mask_discrim_loss = torch.mean((1-mask_discrim_prob[0])**2 + (f_mask_discrim_prob[0])**2)
            discrim_loss = im_discrim_loss + mask_discrim_loss
            discrim_loss.backward()
            disc_optimizer.step()
            print(gen_loss)
            print(discrim_loss)

        if epoch % 10 == 0 and args.show_images:
            image = transforms.ToPILImage()(predicted_image.cpu().detach()[0,:,:,:])
            image.save("./Images/epoch_"+str(epoch)+".png")
        # Saves a checkpoint if needed
        if args.checkpoint_frequency and epoch % args.checkpoint_frequency == 0:
            torch.save({
                'epoch': epoch,
                'gen_loss': gen_loss,
                'discrim_loss': discrim_loss,
                'image_gen_model': image_gen.state_dict(),
                'mask_gen_model': mask_gen.state_dict(),
                'image_disc_model': image_disc.state_dict(),
                'mask_disc_model': mask_disc.state_dict(),
                'gen_optimizer_model': gen_optimizer.state_dict(),
                'disc_optimizer_model': disc_optimizer.state_dict()}, "./checkpoints/epoch_"+str(epoch)+".tar")

if __name__ == "__main__":
    train()
