from competition.encoder import encode, header_bits
from competition.decoder import decode
from competition.cued_sf2_lab.familiarisation import load_mat_img, plot_image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import numpy as np


# Load Image
X, _ = load_mat_img('C2022.mat', img_info='X')


# Encode
vlc, header = encode(X)
size = sum(vlc[:, 1])


# Calculate bits in the header
b = header_bits(header)
print(f"Header bits: {b}")


# Decode
Z = decode(vlc, header)


# Quality indices
print(f"RMS metric: {round(np.std(Z - X), 4)}")
print(f"SSIM metric: {round(ssim(Z, X), 4)}")
print(f"PSNR metric: {round(psnr(X, Z, data_range=256), 4)}")

# Plot reconstructed image
fig, ax = plt.subplots()
plt.grid(False)
plt.axis('off')
plot_image(Z, ax=ax)
plt.savefig("elizabeth_optimal.png")
plt.show()