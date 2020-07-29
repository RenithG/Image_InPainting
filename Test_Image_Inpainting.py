
# This is the Testing code for Image Inpainting with regular / square masking based on GAN.
# To run this code : python Test_Image_Inpainting.py
# Assign the pretrained image inpainting model to TRAINED_MODEL_NAME.
# Provide the path of the test image to TEST_IMAGE macro.

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

INP_IMAGE_SIZE = 128						#input image size
INP_MASK_SIZE = 64							#input mask size
IMG_CHANNELS = 3							#image channel
TRAINED_MODEL_NAME = "inpaint_model.pth"	#name of the pretrained inpainting model
TEST_IMAGE = "Damaged_Image.JPG"			# name of the test image

#Verify whether CUDA is available in machine
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)

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

#Initialize the model function
trained_model = Generator()
trained_model.to(device)
optimizer = torch.optim.Adam(trained_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

#Load the trained model for evaluation
checkpoint = torch.load(TRAINED_MODEL_NAME)
trained_model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
trained_model.eval()

image_transforms = [
    transforms.Resize((INP_IMAGE_SIZE, INP_IMAGE_SIZE), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

#Load the test image to be inpainted
inp_image = Image.open(TEST_IMAGE)
image_transforms = transforms.Compose(image_transforms)
inp_image = image_transforms(inp_image)
inp_image = inp_image.unsqueeze(0)

#Applying the sample regular masking for testing
mask_start_pos = (INP_IMAGE_SIZE - INP_MASK_SIZE) // 2
image_masked = inp_image.clone()
x = int(INP_IMAGE_SIZE / 4)
y = int(INP_IMAGE_SIZE / 4)
range_x = int(x + INP_MASK_SIZE)
range_y = int(y + INP_MASK_SIZE)
image_masked[:, x : range_x, y : range_y] = 1
image_masked = Variable(image_masked.type(torch.cuda.FloatTensor))

# Generate inpainted image from the model
generator_output = trained_model(image_masked)
inpainted_image = image_masked.clone()
inpainted_image[:, :, mask_start_pos : mask_start_pos + INP_MASK_SIZE,
	mask_start_pos : mask_start_pos + INP_MASK_SIZE] = generator_output

#Store the inpainted ouput result
inpainted_image = inpainted_image[0]
img = inpainted_image.clone().add(1).div(2).mul(255).clamp(0, 255).cpu().detach().numpy()
img = img.transpose(1, 2, 0).astype("uint8")
img = Image.fromarray(img)
img.save('Inpainted_Output.png')
