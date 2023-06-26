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
                kernel_size=2,
                latent_dim=16,
                init_channels=16, **kwargs):
        super().__init__()
        # Definir las capas de la red neuronal

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels * 8, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels * 8, out_channels=init_channels * 12, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels * 12, out_channels=256, kernel_size=kernel_size,
            stride=2, padding=0
        )

        self.encoder_hidden_layer = nn.Linear(
            in_features=256, out_features=512  # Ajustar el tamaño de entrada
        )
        self.mean = nn.Linear(
            in_features=512, out_features=8
        )
        self.logvar = nn.Linear(
            in_features=512, out_features=8
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=8, out_features=256  # Ajustar el tamaño de salida
        )

        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=256, out_channels=init_channels * 12, kernel_size=kernel_size,
            stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels * 12, out_channels=init_channels * 8, kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels * 8, out_channels=init_channels * 6 , kernel_size=kernel_size,
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels * 6, out_channels=image_channels, kernel_size=kernel_size,
            stride=2, padding=1
        )

    def encode(self, x):
        activation = torch.relu(self.enc1(x))
        activation = torch.relu(self.enc2(activation))
        activation = torch.relu(self.enc3(activation))
        activation = torch.relu(self.enc4(activation))

        activation = F.adaptive_avg_pool2d(activation, 1).squeeze()
        activation = self.encoder_hidden_layer(activation)
        mean = self.mean(activation)
        logvar = self.logvar(activation)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, z):
        activation = self.decoder_hidden_layer(z)
        activation = activation.view(activation.size(0), 256, 1, 1)  # Ajustar el tamaño
        activation = torch.relu(self.dec1(activation))
        activation = torch.relu(self.dec2(activation))
        activation = torch.relu(self.dec3(activation))
        reconstructed = torch.sigmoid(self.dec4(activation))
        reconstructed = reconstructed .view(-1, 1024, 1, 1)
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
    root_dir = 'F:/URJC/TFG/Texturas/Minecraft/Texturas/1AADATASET'

    # Transformaciones opcionales que puedes aplicar a las imágenes
    transform = torchvision.transforms.Compose([
        Resize((16, 16)),  # Cambiar el tamaño a 32x32
        ToTensor()
    ])

    # Crear instancia del conjunto de datos personalizado
    dataset = CustomDataset(root_dir, transform)

    # Crear un cargador de datos para iterar sobre los lotes de datos
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(input_shape=16 * 16 * 4).to(device)  # Ajustar el tamaño de entrada
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum')  # Cambiar a 'sum' en lugar de 'size_average'

    def loss_function(recon_x, x, mu, logvar):
        BCE = criterion(recon_x, x.view(-1, 16 * 16 * 4))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    for epoch in range(epochs):
        train_loss = 0
        for batch_features, _ in train_loader:
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            outputs, mean, logvar = model(batch_features)
            loss = loss_function(outputs, batch_features, mean, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        train_loss /= len(train_loader)

        print("Epoch : {}/{}, Loss: {:.8f}".format(epoch + 1, epochs, train_loss))

    torch.save(model.state_dict(), 'modelo_vae_Minecraft_convo.pth')
    print("Modelo guardado correctamente.")

    with torch.no_grad():
        latent = torch.randn(10, 8).to(device)
        generated_samples = model.decode(latent)

    generated_samples = generated_samples.cpu().view(-1, 4, 16, 16)  # Ajustar el número de canales
    generated_samples = np.clip(generated_samples, 0, 1)  # Escalar las muestras al rango [0, 1]

    plt.figure(figsize=(10, 2))
    for i in range(10):
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(np.transpose(generated_samples[i].numpy(), (1, 2, 0)))  # Cambiar el orden de los canales
        plt.axis('off')
    plt.show()
