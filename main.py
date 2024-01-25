import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy

N = 100
N_image = int(0.9*N*N)

image = 'lion.png'
style = 'picasso.png'

image = Image.open(image)
style = Image.open(style)

image_rgb = image.convert('RGB')
style_rgb = style.convert('RGB')

transform = transforms.ToTensor()
image_tensor = transform(image_rgb)
style_tensor = transform(style_rgb)

weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(3, 1, 1)

gray_image = torch.sum(weights * image_tensor, dim=0)
gray_image.squeeze()
gray_style = torch.sum(weights * style_tensor, dim=0)
gray_style.squeeze()


flat_image = torch.flatten(gray_image)
flat_image = torch.unsqueeze(flat_image, dim=1)

flat_style = torch.flatten(gray_style)
flat_style = torch.unsqueeze(flat_style, dim=1)

print("Computing Laplacian")
Degree = torch.diag(torch.ones(N*N))
Adj = 8 * torch.ones(N*N, N*N)

for i in range(N):
    for j in range(N):
        current_node = i * N + j

        
        if i > 0:
            Adj[current_node, (i - 1) * N + j] = 1
        if i < N - 1:
            Adj[current_node, (i + 1) * N + j] = 1
        if j > 0:
            Adj[current_node, i * N + (j - 1)] = 1
        if j < N - 1:
            Adj[current_node, i * N + (j + 1)] = 1

Laplacian = Degree - Adj

print("Computing eig")
eigenvalues, eigenvectors = numpy.linalg.eigh(Laplacian.numpy())
eigenvalues, eigenvectors = torch.tensor(eigenvalues), torch.tensor(eigenvectors)
print("sorting")

indices = torch.argsort(eigenvalues.abs(), descending=True)

sorted_eigenvalues = eigenvalues[indices]
sorted_eigenvectors = U = eigenvectors[:, indices]

print("done")

spectral_image = torch.mm(sorted_eigenvectors, flat_image)
spectral_image[N_image:] = 0
spectral_style = torch.mm(sorted_eigenvectors, flat_style)
spectral_style[:N_image] = 0

output = torch.mm(sorted_eigenvectors.transpose(0, 1),
                  spectral_image + spectral_style)
output = output.reshape(N, N)

plt.imshow(output.numpy(), cmap='viridis')
plt.colorbar()
plt.title('Heatmap de la matrice')
plt.xlabel('Axe X')
plt.ylabel('Axe Y')

plt.show()
