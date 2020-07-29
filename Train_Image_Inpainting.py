
# This is the Training code for Image Inpainting with regular / square masking based on GAN.
# To run this code : python Train_Image_Inpainting.py
# Keep the training images under DATASET_NAME folder.

# Importing neccessary python package
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import glob
import random

NUM_EPOCHS = 200					#total number of epochs for network training
BATCH_SIZE = 8						#value of batch size
DATASET_NAME = "paris_street_view"	#place the training images in this folder
INP_IMAGE_SIZE = 128				#input image size
INP_MASK_SIZE = 64					#input mask size
IMG_CHANNELS = 3					#image channel
SAMPLE_INTERVAL = 1000				#output image sampling period

#Create folder for saving sampled reconstructed output images
if not os.path.exists('output_images'):
    os.makedirs('output_images')

#Verify whether CUDA is available in machine
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

#Initializing weights for the model 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Initializing the Loss Function
criterionMSE = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
criterionMSE.to(device)
criterion.to(device)

#Constructing Generator Model 
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.generator_model = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2),            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2),            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2),             
            nn.Conv2d(512, 4000, 1),
            nn.ConvTranspose2d(4000, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512, 0.8),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(),            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(),             
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(), 
            nn.Conv2d(64, IMG_CHANNELS, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, data):
        return self.generator_model(data)

#Constructing Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator_model = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, image_data):
        return self.discriminator_model(image_data)

#Initialize Generator
generator = Generator()
generator.to(device)

#Initialize Discriminator
discriminator = Discriminator()
discriminator.to(device)

#Applying Initial Weight to Generator and Discriminator
generator.apply(weights_init)
discriminator.apply(weights_init)

#Class for loading, processing and masking the input dataset
class InpaintDatasetLoader(Dataset):
    def __init__(self, dataset_path, image_transforms=None,
	inp_imgsize=128, inp_masksize=64, is_Trainable=True):
        self.image_transforms = transforms.Compose(image_transforms)
        self.inp_imgsize = inp_imgsize
        self.inp_masksize = inp_masksize
        self.is_Trainable = is_Trainable
        self.input_data = glob.glob(dataset_path + "/*.JPG")
        self.input_data = sorted(self.input_data)
        if is_Trainable == True:
            self.input_data = self.input_data[:-3000]
        else:
            self.input_data = self.input_data[-3000:]

    # Function for Random regular masking 
    def random_regular_mask(self, inp_image):
        image_masked = inp_image.clone()
        y, x = np.random.randint(0, self.inp_imgsize - self.inp_masksize, 2)
        range_x = int(x + self.inp_masksize)
        range_y = int(y + self.inp_masksize)
        mask_region = inp_image[:, y:range_y, x:range_x]
        image_masked[:, y:range_y, x:range_x] = 1

        return image_masked, mask_region

    # Function for center area masking 	
    def center_mask(self, inp_image):
        mask_coord_pos = (self.inp_imgsize - self.inp_masksize) // 2
        image_masked = inp_image.clone()
        x = int(self.inp_imgsize / 4)
        y = int(self.inp_imgsize / 4)
        range_x = int(x + self.inp_masksize)
        range_y = int(y + self.inp_masksize)
        image_masked[:, x : range_x, y : range_y] = 1

        return image_masked, mask_coord_pos

    def __getitem__(self, index):

        inp_image = Image.open(self.input_data[index % len(self.input_data)])
        inp_image = self.image_transforms(inp_image)
        if self.is_Trainable == True:
            # Random regular masking for training images
            image_masked, mask_region = self.random_regular_mask(inp_image)
        else:
            # Central area masking for testing images
            image_masked, mask_region = self.center_mask(inp_image)

        return inp_image, image_masked, mask_region

    def __len__(self):
        return len(self.input_data)

#DataLoader for train and test data
image_transforms = [
    transforms.Resize((INP_IMAGE_SIZE, INP_IMAGE_SIZE), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    InpaintDatasetLoader("input_dataset/%s" % DATASET_NAME,
	image_transforms=image_transforms, is_Trainable=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)
print('Length of dataloader is : ',len(dataloader))

test_dataloader = DataLoader(
    InpaintDatasetLoader("input_dataset/%s" % DATASET_NAME,
	image_transforms=image_transforms, is_Trainable=False),
    batch_size=12,
    shuffle=True,
    num_workers=1,
)
print('Length of test dataloader is : ',len(test_dataloader))

#Initializing Optimizer for Generator and Discriminator
generator_optimizer = torch.optim.Adam(generator.parameters(),
						lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
						lr=0.0002, betas=(0.5, 0.999))

#Visualizing Sample Output Images
def store_sample_image(check_sample_count):
    inp_image, image_masked, mask_coord = next(iter(test_dataloader))
    inp_image = Variable(inp_image.type(torch.cuda.FloatTensor))
    image_masked = Variable(image_masked.type(torch.cuda.FloatTensor))
    mask_coord = mask_coord[0].item()  
    # Output of reconstructed generated image
    generator_output = generator(image_masked)
    inpainted_image = image_masked.clone()
    inpainted_image[:, :, mask_coord : mask_coord + INP_MASK_SIZE,
					mask_coord : mask_coord + INP_MASK_SIZE] = generator_output
    # Store the output data to disk
    sample = torch.cat((image_masked.data, inpainted_image.data, inp_image.data), -2)
    save_image(sample, "output_images/%d.png" % check_sample_count, nrow=6, normalize=True)

#Training the inpainting network
for epoch in range(NUM_EPOCHS):
    for i, (inp_image, image_masked, mask_region) in enumerate(dataloader):

        # Setting real / fake class
        real_labels = Variable(
		torch.cuda.FloatTensor(
		inp_image.shape[0], 1,int(INP_MASK_SIZE / 2 ** 3), int(INP_MASK_SIZE / 2 ** 3)
		).fill_(1.0), requires_grad=False
		)
        fake_labels = Variable(
		torch.cuda.FloatTensor(
		inp_image.shape[0], 1,int(INP_MASK_SIZE / 2 ** 3), int(INP_MASK_SIZE / 2 ** 3)
		).fill_(0.0), requires_grad=False
		)

        # Setting input image data 
        inp_image = Variable(inp_image.type(torch.cuda.FloatTensor))
        image_masked = Variable(image_masked.type(torch.cuda.FloatTensor))
        mask_region = Variable(mask_region.type(torch.cuda.FloatTensor))

        # Training Generator
        generator_optimizer.zero_grad()

        # Generate images from the generator 
        generator_output = generator(image_masked)

        # Calculate Loss from the Generator side
        gen_entropy = criterionMSE(discriminator(generator_output), real_labels)
        gen_reconstruct = criterion(generator_output, mask_region)
        generator_loss = 0.001 * gen_entropy + 0.999 * gen_reconstruct

        generator_loss.backward()
        generator_optimizer.step()

        # Training Discriminator
        discriminator_optimizer.zero_grad()

        # Classifying real / fake images from the samples generated
        real_loss = criterionMSE(discriminator(mask_region), real_labels)
        fake_loss = criterionMSE(discriminator(generator_output.detach()), fake_labels)
        discriminator_loss = (real_loss + fake_loss)/2

        discriminator_loss.backward()
        discriminator_optimizer.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, NUM_EPOCHS, i, len(dataloader), discriminator_loss.item(),
			generator_loss.item())
        )

        # Saving generated sample at some sample interval
        check_sample_count = epoch * len(dataloader) + i
        if check_sample_count % SAMPLE_INTERVAL == 0:
            store_sample_image(check_sample_count)
            #save the model checkpoint here
            checkpoint = {
                'state_dict': generator.state_dict(),
                'optimizer': generator_optimizer.state_dict(),
            }
            torch.save(checkpoint, "inpaint_model.pth")

print ('Training Ends !!!')

