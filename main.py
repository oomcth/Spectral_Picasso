import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy

# image size
N = 100
N_image = int(0.9*N*N)

# Path to your image file
image = 'lion.png'
style = 'picasso.png'

# Load the image using PIL
image = Image.open(image)
style = Image.open(style)

# Convert the image to RGB if it's not already
image_rgb = image.convert('RGB')
style_rgb = style.convert('RGB')

# Transform to a PyTorch tensor
transform = transforms.ToTensor()
image_tensor = transform(image_rgb)
style_tensor = transform(style_rgb)

# Split the tensor into three channels (Red, Green, Blue)
# image_r, image_g, image_b = image_tensor.split(1, dim=0)
# style_r, style_g, style_b = style_tensor.split(1, dim=0)

# Poids pour la conversion en niveaux de gris
weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)

# Convertir en niveaux de gris
gray_image = torch.sum(weights * image_tensor, dim=0)
gray_image.squeeze()
gray_style = torch.sum(weights * style_tensor, dim=0)
gray_style.squeeze()

f = torch.flatten(gray_style).reshape(N, N)
plt.imshow(gray_image.numpy(), cmap='viridis')  # cmap est la colormap que vous souhaitez utiliser
plt.colorbar()  # Ajoute une barre de couleur pour la légende
plt.title('Heatmap de la matrice')  # Titre du graphique
plt.xlabel('Axe X')  # Libellé de l'axe X
plt.ylabel('Axe Y')  # Libellé de l'axe Y

# Affichez le graphique
plt.show()

flat_image = torch.flatten(gray_image)
flat_image = torch.unsqueeze(flat_image, dim=1)

flat_style = torch.flatten(gray_style)
flat_style = torch.unsqueeze(flat_style, dim=1)

print("Computing Laplacian")
Degree = torch.diag(torch.ones(N*N))
Adj = 8 * torch.ones(N*N, N*N)

for i in range(N):
    for j in range(N):
        current_node = i * N + j  # Index du nœud actuel dans la matrice d'adjacence

        # Ajoutez des connexions aux pixels voisins (distance de 1).
        if i > 0:
            Adj[current_node, (i - 1) * N + j] = 1  # Voisin du haut
        if i < N - 1:
            Adj[current_node, (i + 1) * N + j] = 1  # Voisin du bas
        if j > 0:
            Adj[current_node, i * N + (j - 1)] = 1  # Voisin de gauche
        if j < N - 1:
            Adj[current_node, i * N + (j + 1)] = 1  # Voisin de droite

Laplacian = Degree - Adj

print("Computing eig")
eigenvalues, eigenvectors = numpy.linalg.eigh(Laplacian.numpy())
eigenvalues, eigenvectors = torch.tensor(eigenvalues), torch.tensor(eigenvectors)
print("sorting")

indices = torch.argsort(eigenvalues.abs(), descending=True)

sorted_eigenvalues = eigenvalues[indices]
sorted_eigenvectors = U = eigenvectors[:, indices]

print("done")

# temp = torch.mm(sorted_eigenvectors, sorted_eigenvectors.transpose(0, 1))

# plt.imshow(temp.numpy(), cmap='viridis')  # cmap est la colormap que vous souhaitez utiliser
# plt.colorbar()  # Ajoute une barre de couleur pour la légende
# plt.title('Heatmap de la matrice')  # Titre du graphique
# plt.xlabel('Axe X')  # Libellé de l'axe X
# plt.ylabel('Axe Y')  # Libellé de l'axe Y

# Affichez le graphique
# plt.show()

spectral_image = torch.mm(sorted_eigenvectors, flat_image)
spectral_image[N_image:] = 0
spectral_style = torch.mm(sorted_eigenvectors, flat_style)
spectral_style[:N_image] = 0

output = torch.mm(sorted_eigenvectors.transpose(0, 1),
                  spectral_image + spectral_style)
output = output.reshape(N, N)

plt.imshow(output.numpy(), cmap='viridis')  # cmap est la colormap que vous souhaitez utiliser
plt.colorbar()  # Ajoute une barre de couleur pour la légende
plt.title('Heatmap de la matrice')  # Titre du graphique
plt.xlabel('Axe X')  # Libellé de l'axe X
plt.ylabel('Axe Y')  # Libellé de l'axe Y

# Affichez le graphique
plt.show()
