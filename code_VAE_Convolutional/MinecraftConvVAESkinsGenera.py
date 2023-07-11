import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGBA")

        if self.transform:
            image = self.transform(image)

        return image, 0  # Puedes reemplazar 0 con cualquier información adicional que desees proporcionar

class VAE(nn.Module):
    def __init__(self, image_channels=4,
                kernel_size=4,
                latent_dim=16,
                init_channels=16, **kwargs):
        super().__init__()
        # Definir las capas de la red neuronal

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=32, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=kernel_size,
            stride=2, padding=1
        )


        self.mean = nn.Linear(
            in_features=256*4*4, out_features=100
        )
        self.logvar = nn.Linear(
            in_features=256*4*4, out_features=100
        )


        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=100, out_channels=256, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec5 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec6 = nn.ConvTranspose2d(
            in_channels=16, out_channels=4, kernel_size=kernel_size,
            stride=2, padding=1
        )

    def encode(self, x):
        activation = torch.relu(self.enc1(x))
        activation = torch.relu(self.enc2(activation))
        activation = torch.relu(self.enc3(activation))
        activation = torch.relu(self.enc4(activation))

        activation = activation.view(activation.size(0), -1)
        mean = self.mean(activation)
        logvar = self.logvar(activation)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, z):
        activation = z.view(z.size(0), z.size(1), 1, 1)
        activation = torch.relu(self.dec1(activation))
        activation = torch.relu(self.dec2(activation))
        activation = torch.relu(self.dec3(activation))
        activation = torch.relu(self.dec4(activation))
        activation = torch.relu(self.dec5(activation))
        reconstructed = torch.sigmoid(self.dec6(activation))
        return reconstructed

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE().to(device)  # Ajustar el tamaño de entrada
    model.load_state_dict(torch.load('modelo_vae_Minecraft_convo_skins_3.pth'))
    model.eval()

    print("Modelo cargado correctamente.")
    with torch.no_grad():
        latent = torch.randn(200, 100).to(device)
        generated_samples = model.decode(latent)

    generated_samples = generated_samples.cpu().view(-1, 4, 64, 64)  # Ajustar el número de canales
    generated_samples = np.clip(generated_samples, 0, 1)  # Escalar las muestras al rango [0, 1]

    os.makedirs("generated_images", exist_ok=True)  # Crear la carpeta para las imágenes generadas

    for i in range(200):
        generated_image = generated_samples[i]
        file_name = f"generated_image_2_{i}.png"
        file_path = os.path.join("generated_images_convo_skins", file_name)
        torchvision.utils.save_image(generated_image, file_path)

    print("Imágenes generadas guardadas correctamente.")

