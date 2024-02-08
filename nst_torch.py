import torch
from torch import nn
from torch import optim
from torchvision import models
from torchvision import transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime
import pytz


class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    self.chosen_features = ['0', '5', '10', '19', '28']
    self.model = models.vgg19(weights=True).features[:29]

  def forward(self, x):
    features = []
    for layer_num, layer in enumerate(self.model):
      x = layer(x)
      if str(layer_num) in self.chosen_features:
        features.append(x)

    return features

# Hyper parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
steps = 6000
lr = 5e-3
alpha = 5
beta = 1e-3
image_size = 256


# Load for processing
loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# Inverse processed tensor
inv_loader = transforms.Compose([
    transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])
])

# Load image function for input
def load_image(image_name):
  image = Image.open(image_name)
  image = loader(image).unsqueeze(0)
  return image.to(device)


# Compute loss functions
def compute_content_loss(generated_matrix, content_matrix):
  return torch.mean((generated_matrix - content_matrix)**2)

def compute_style_loss(generated_matrix, style_matrix):
  # Gram matrix
  G = generated_matrix.view(channels, height*width).mm(generated_matrix.view(channels, height*width).t())
  S = style_matrix.view(channels, height*width).mm(style_matrix.view(channels, height*width).t())

  return torch.mean((G - S)**2)

# Create folder to store images
def create_folders():
  # Get time parts
  vietnam_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
  current_time = datetime.now(vietnam_timezone)
  time_part = current_time.strftime("_%Y-%m-%d_%H-%M")

  # Create parent part
  parent_part = "Gans_Torch"
  if not os.path.exists(parent_part):
    os.mkdir(parent_part)

  # Main part
  directory = os.path.join('Gans_Torch', f"outputs{time_part}")
  if not os.path.exists(directory):
    os.mkdir(directory)

  return directory


def plot_images_and_save(content_path, style_path):

  # Open content and style image
  content_image = Image.open(content_path)
  style_image = Image.open(style_path)

  # Transform Resize
  resize_loader = transforms.Compose([
      transforms.Resize((image_size, image_size))
  ])

  # Transform ToPILImage
  convert_loader = transforms.Compose([
      transforms.ToPILImage()
  ])

  content_image = resize_loader(content_image)
  style_image = resize_loader(style_image)

  # Create fig
  fig, axs = plt.subplots(1, 3, figsize=(21, 7))

  # Content image
  axs[0].imshow(content_image)
  axs[0].set_title('Content Image')
  axs[0].axis('off')

  # Style image
  axs[1].imshow(style_image)
  axs[1].set_title('Style Image')
  axs[1].axis('off')

  # Generated image process
  generated = generated_image.to('cpu').detach().numpy().squeeze(0).transpose(1, 2, 0)
  generated = (generated * 255).astype('uint8')
  generated = convert_loader(generated)

  # Generated image
  axs[2].imshow(generated)
  axs[2].set_title('Generated Image')
  axs[2].axis('off')

  # Save last fig
  last_path = os.path.join(directory, "final_result.png")
  plt.savefig(last_path)

  # Show fig
  plt.show()


# Main training loop
# Create model instance
model = VGG().to(device)

# Define things
content_path = "Gans_Torch/Assets/content1.jpg"
style_path = "Gans_Torch/Assets/style1.jpg"
save_image_interval = 10
print_loss_interval = 10
steps = 2000

# Define images
content_image = load_image(content_path)
style_image = load_image(style_path)
generated_image = content_image.clone().requires_grad_(True)

# Define optimizer
optimizer = optim.Adam([generated_image], lr=lr)

# Create folder
directory = create_folders()

total_loss = 0
loss_path = os.path.join(directory, "loss.txt")

with open(loss_path, "w") as f:
  # Main loop
  for step in range(1, steps+1):
    # Get features
    style_features = model(style_image)
    content_features = model(content_image)
    generated_features = model(generated_image)

    style_loss = content_loss = 0

    # Compute losses
    for style_feature, content_feature, generated_feature in zip(style_features, content_features, generated_features):
      batch_size, channels, height, width = generated_feature.shape
      content_loss += compute_content_loss(generated_feature, content_feature)
      style_loss += compute_style_loss(generated_feature, style_feature)
    total_loss = alpha * content_loss + beta * style_loss

    # Gradient
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Print loss after 20 steps
    if step % print_loss_interval == 0:
      print(f"Step {step}/{steps} Loss = {total_loss.item()}")
      f.write(f"Step {step}/{steps} Loss = {total_loss.item()}\n")

    # Save image after 200 steps
    if (step % save_image_interval == 0):
      print(f"Saving...")

      # Save generated images to target directory in Drive
      file_path = os.path.join(directory, f"({step}).png")
      save_image(generated_image, file_path)
      print(f"generated_{step}.png saved.")
