import os
import glob
from PIL import Image
import torch
from torchvision import models, transforms
from torch.linalg import norm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

lst_colors = [
    # '#f9dbbd',
    "#ffffff",
    "#e1e5f2",
    '#bfdbf7',
    "#4ea8de",
    '#219ebc',
    '#022b3a',
    # '#450920',
]
from matplotlib.colors import LinearSegmentedColormap
custom1 = LinearSegmentedColormap.from_list(
    name='pink',
    colors=lst_colors,
)

class LinearCKA:
    def __init__(self, device='cuda'):
        self.device = device

    def linear_HSIC(self, X, Y, sigma=None):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def centering(self, K):
        n = K.shape[0]
        self.H = self.get_centering_matrix(n)
        return torch.matmul(torch.matmul(self.H, K), self.H)

    def get_centering_matrix(self, n):
        unit = torch.ones(n, n).to(self.device)
        I = torch.eye(n).to(self.device)
        H = I - unit / n
        return H

    def calculate(self, X, Y, sigma=None):
        hsic = self.linear_HSIC(X, Y, sigma)
        # print(hsic)
        # asd
        var1 = torch.sqrt(self.linear_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)


# Load pre-trained ResNet18 model and set it to evaluation mode
model = models.resnet18(pretrained=True)
model.eval()
modules = list(model.children())[:-3]  # Remove the last few layers to extract intermediate features
resnet18_features = nn.Sequential(*modules)

# Define image transformation: resize, convert to tensor, and normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Function to load and process image
def load_and_process_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert image to RGB
    return transform(image).unsqueeze(0)  # Add batch dimension

image_folder = "/volumes1/datasets/DN4IL/tmp/ice"
image_files = []
domains = ['real', 'clipart', 'infograph', 'painting', 'sketch', 'quickdraw']
all_files = os.listdir(image_folder)

for dom in domains:
    domain_files = [os.path.join(image_folder, f) for f in all_files if f.startswith(dom) and f.endswith('.jpg')]
    image_files.extend(domain_files)
    # image_files.append(os.path.join(image_folder, '{}_*.jpg'.format(dom)))
images = [load_and_process_image(path) for path in image_files]

# Extract features and flatten them
features = []
for image in images:
    with torch.no_grad():
        feature = resnet18_features(image).cuda() # Extract intermediate features
        feature = feature.view(feature.size(0), feature.size(1), -1) #feature.view(feature.size(0), -1)  # Flatten (batch_size, channels * height * width)
        feature = feature.mean(dim=2).permute(1,0)
        features.append(feature)  # Remove the batch dimension

cka = LinearCKA()
similarity_matrix = torch.zeros((6, 6))
for i in range(6):
    for j in range(6):
        similarity_matrix[i, j] = cka.calculate(features[i], features[j])


masked_similarity_matrix = np.tril(similarity_matrix.numpy())  # Convert to NumPy array and get lower triangle
print(masked_similarity_matrix)
# Plot the CKA similarity matrix
fig, ax = plt.subplots()
plt.imshow(masked_similarity_matrix, cmap=custom1, interpolation='nearest')
plt.colorbar()
plt.title("CKA Similarity between Images")
domain_label = ['Real', 'Clipart', 'Infograph', 'Painting', 'Sketch', 'Quickdraw']
plt.xticks(range(6), domain_label, rotation=45)
plt.yticks(range(6), domain_label, rotation=45)
# plt.savefig("/volumes1/vlm-cl/paper/cka_ice.png", bbox_inches='tight')
plt.show()
