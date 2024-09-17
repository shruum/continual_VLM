import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from PIL import Image
matplotlib.use('TkAgg')
# Convert the tensor to a NumPy array
# Transpose the image from (3, 64, 64) to (64, 64, 3)
first_image_np = np.transpose(data[0].cpu().numpy(), (1, 2, 0))
# Assuming mean and std used for normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
# Denormalize the image
first_image_np = std * first_image_np + mean
first_image_np = np.clip(first_image_np, 0, 1)
# Convert NumPy array to PIL image
first_image_pil = Image.fromarray((first_image_np * 255).astype(np.uint8))
# Resize the image to a higher resolution (e.g., 256x256)
first_image_resized = first_image_pil.resize((256, 256))
# Plot the image
plt.imshow(first_image_resized)
plt.axis('on')  # Hide the axis
plt.show()
# first_image_resized.save('tmp.jpg')