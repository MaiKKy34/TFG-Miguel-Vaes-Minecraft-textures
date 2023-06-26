import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
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
    def __init__(self, **kwargs):
        super().__init__()
        # Definir las capas de la red neuronal
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        self.encoder_hidden_layer_2 = nn.Linear(
            in_features=512, out_features=512
        )
        self.mean = nn.Linear(
            in_features=512, out_features=8
        )
        self.logvar = nn.Linear(
            in_features=512, out_features=8
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=8, out_features=512
        )
        self.decoder_hidden_layer_2 = nn.Linear(
            in_features=512, out_features=512
        )
        self.decoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )

    def encode(self, x):
        activation = self.encoder_hidden_layer(x)
        activation = torch.relu(activation)
        activation = self.encoder_hidden_layer_2(activation)
        activation = torch.sigmoid(activation)
        mean = self.mean(activation)
        logvar = self.logvar(activation)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, z):
        activation = self.decoder_hidden_layer(z)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer_2(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(input_shape=64 * 64 * 4).to(device)  # Ajustar el tamaño de entrada
    model.load_state_dict(torch.load('modelo_vae_Minecraft_skins.pth'))
    model.eval()

    print("Modelo cargado correctamente.")
    with torch.no_grad():
        latent = torch.randn(200, 8).to(device)
        generated_samples = model.decode(latent)

    generated_samples = generated_samples.cpu().view(-1, 4, 64, 64)  # Ajustar el número de canales
    generated_samples = np.clip(generated_samples, 0, 1)  # Escalar las muestras al rango [0, 1]

    os.makedirs("generated_images", exist_ok=True)  # Crear la carpeta para las imágenes generadas

    for i in range(200):
        generated_image = generated_samples[i]
        file_name = f"generated_image_skins_1_{i}.png"
        file_path = os.path.join("generated_images_skins", file_name)
        torchvision.utils.save_image(generated_image, file_path)

    print("Imágenes generadas guardadas correctamente.")