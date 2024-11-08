import os
from PIL import Image
import torch
from torchvision import models, transforms
from torch.linalg import norm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer

lst_colors = [
    # "#ffffff",
    "#bfdbf7",
    "#e1e5f2",
    '#bfdbf7',
    "#4ea8de",
    '#219ebc',
    '#022b3a',
]
from matplotlib.colors import LinearSegmentedColormap
custom1 = LinearSegmentedColormap.from_list(
    name='blue_custom',
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


# text = ("An airplane is a vehicle with 2 wings on the side and motors."
#         " It can be real photo or a drawn sketch and it can be colorful or grey. It can be painted") # is a sleek, metallic body with swept-back wings, a pointed nose, and engines attached to the wings and tail section."
# text = "Rough sketch of an object which is black and white"
# text = "Blue skies with white clouds"


def get_penultimate_embedding(model, text):
    tokenized_text = model.tokenize([text])
    tokenized_text['attention_mask'] = tokenized_text['attention_mask'].to('cuda')
    tokenized_text['input_ids'] = tokenized_text['input_ids'].to('cuda')
    transformer_model = model[0].auto_model  # Correct way to access the Hugging Face transformer
    with torch.no_grad():
        outputs = transformer_model(input_ids=tokenized_text['input_ids'], attention_mask=tokenized_text['attention_mask'],output_hidden_states=True)
        penultimate_layer_embedding = outputs.hidden_states[-1]  # Second to last layer
    return penultimate_layer_embedding #.mean(dim=0)


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
# Image file paths (modify this path as needed)
# text = "An airplane is a vehicle with 2 wings on the side and motors and it can be painting with blue and green color or a black sketch"
text = "An ice cream is a frozen, creamy, and colorful treat atop a cone or in a cup, and may have various toppings"
image_folder = "/volumes1/datasets/DN4IL/tmp/ice"
image_files = []
domains = ['real', 'clipart', 'infograph', 'painting', 'sketch', 'quickdraw']

all_files = os.listdir(image_folder)
for dom in domains:
    domain_files = [os.path.join(image_folder, f) for f in all_files if f.startswith(dom) and f.endswith('.jpg')]
    image_files.extend(domain_files)
images = [load_and_process_image(path) for path in image_files]

# Extract features and flatten them
features = []
for image in images:
    with torch.no_grad():
        feature = resnet18_features(image).cuda()  # Extract intermediate features
        feature = feature.view(feature.size(0), feature.size(1), -1) #feature.view(feature.size(0), -1)  # Flatten (batch_size, channels * height * width)
        feature = feature.mean(dim=2)
        image_features_padded = torch.nn.functional.pad(feature, (0,128)).permute(1,0) # Padding along the second dimension
        features.append(image_features_padded)  # Store the features

# Sentence to extract embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models as well
text_embedding = get_penultimate_embedding(sentence_model, text)
text_embedding_penultimate = text_embedding.mean(dim=1)
text_embedding_penultimate = text_embedding_penultimate.permute(1, 0)
# min_val = text_embedding_penultimate.min()  # Get the minimum value in the tensor
# max_val = text_embedding_penultimate.max()  # Get the maximum value in the tensor
# # Apply min-max normalization
# epsilon = 1e-7
# text_embedding_penultimate = (text_embedding_penultimate - min_val) / (max_val - min_val + epsilon)
# text_embedding_penultimate = text_embedding_penultimate


cka = LinearCKA()
similarity_matrix = torch.zeros(1, 6)
for i in range(6):
    similarity_matrix[0, i] = cka.calculate(features[i], text_embedding_penultimate)

print(similarity_matrix)
min_val = similarity_matrix.min()
max_val = similarity_matrix.max()
normalized_similarity_matrix = (similarity_matrix - min_val) / (max_val - min_val)

# Plot the 1x5 CKA similarity matrix
fig, ax = plt.subplots(figsize=(6,1))
plt.imshow(normalized_similarity_matrix, cmap=custom1, interpolation='nearest', aspect='auto')
# plt.colorbar()
plt.title("CKA Similarity between Images and Text")
domain_label = ['Real', 'Clipart', 'Infograph', 'Painting', 'Sketch', 'Quickdraw']
plt.xticks(range(6), domain_label, rotation=20)
plt.yticks([])  # No y-axis labels needed since it's only one row
plt.savefig("/volumes1/vlm-cl/paper/cka_text_ice.png", bbox_inches='tight')
plt.show()
