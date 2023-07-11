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
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batch_size = 128
    epochs = 100
    learning_rate = 1e-3

    # Ruta a la carpeta que contiene tus imágenes de entrenamiento
    root_dir = 'F:/URJC/TFG/Texturas/Minecraft/Skins/Buenisimas'

    # Transformaciones opcionales que puedes aplicar a las imágenes
    transform = torchvision.transforms.Compose([
        Resize((64, 64)),
        ToTensor()
    ])

    # Crear instancia del conjunto de datos personalizado
    dataset = CustomDataset(root_dir, transform)

    # Crear un cargador de datos para iterar sobre los lotes de datos
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(input_shape=64 * 64 * 4).to(device)  # Ajustar el tamaño de entrada
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum')  # Cambiar a 'sum' en lugar de 'size_average'

    def loss_function(recon_x, x, mu, logvar):
        BCE = criterion(recon_x, x.view(-1, 64 * 64 * 4))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    for epoch in range(epochs):
        train_loss = 0
        for batch_features, _ in train_loader:
            batch_features = batch_features.view(-1, 64 * 64* 4).to(device)  # Ajustar el tamaño de entrada
            optimizer.zero_grad()
            outputs, mean, logvar = model(batch_features)
            loss = loss_function(outputs, batch_features, mean, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_loader)

        print("Epoch : {}/{}, Loss: {:.8f}".format(epoch + 1, epochs, train_loss))

    torch.save(model.state_dict(), 'modelo_vae_Minecraft_skins.pth')
    print("Modelo guardado correctamente.")