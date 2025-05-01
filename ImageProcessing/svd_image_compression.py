# Grayscale Image Compression Using SVD
# Kiumbura N. Githinji (s1358017)
# Max (s1337039)
# Adnan Hoti (s1320619)
# MA 221 01

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
# Conversion of RGB Image to Grayscale Image

# Load the color image
img = cv2.imread('flower.jpg')

# Error Handling
if img is None:
    print("Error: Image not found or path is incorrect.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show image
cv2.imshow('Grayscale', gray)
print("Press 'q' to close the image window.")

# Loop until 'q' is pressed
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
"""

# Grayscale Image Compression Using SVD

image = 'flower.jpg'
image_name = 'Flower'

# Load grayscale image as 2D array
img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found.")
    exit()

"""
    # Singular Compression

# Show original image
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original ' + image_name)
plt.axis('off')

# Apply SVD
U, S, VT = np.linalg.svd(img, full_matrices = False)

# Max k (full rank)
k_max = len(S)
print(f"Original number of singular values (k max): {k_max}")

plt.title(f'Original {image_name}\nk max = {k_max}')***

# Choose how many singular values to keep (e.g., 50 out of possible 512)
k = 50  # adjust this to control compression

# Reconstruct the image using only top-k singular values
compressed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), VT[:k, :]))

# Show compressed image
plt.subplot(1, 2, 2)
plt.imshow(compressed, cmap='gray')
plt.title(image_name + f' Compressed (k={k})')
plt.axis('off')

"""
    # Multiple Compression & Multiple Compression Comparison

# Apply SVD once
U, S, VT = np.linalg.svd(img, full_matrices=False)

# Max k (full rank)
k_max = len(S)
plt.title(f'Original {image_name}\nk max = {k_max}')

# Compression levels to compare
k_values = [10, 50, 100]

"""# X Compression
# Set up subplot grid (original + compressed images)
plt.figure(figsize=(15, 5))
plt.subplot(1, len(k_values) + 1, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Original {image_name}\nk max = {k_max}')
plt.axis('off')

# Display each compressed version
for i, k in enumerate(k_values):
    # Reconstruct compressed image
    compressed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), VT[:k, :]))

    # Calculate % of data retained
    compression_ratio = (k / k_max) * 100
    compression_ratio_str = f"{compression_ratio:.1f}%"

    # Plot
    plt.subplot(1, len(k_values) + 1, i + 2)
    plt.imshow(compressed, cmap='gray')
    plt.title(f'{image_name}\nk={k} ({compression_ratio_str})')
    plt.axis('off')

"""
""" # Comparison
# Set up 2 rows (Good vs. Bad) for each k value
plt.figure(figsize=(15, 6))
plt.suptitle(f"{image_name} - Good vs Bad Compression (SVD)", fontsize=14)

# Original image (top-left)
plt.subplot(2, len(k_values) + 1, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Original\nk max = {k_max}')
plt.axis('off')

# Loop over k values
for i, k in enumerate(k_values):
    pct = (k / k_max) * 100

    # Good compression
    compressed = np.dot(U[:, :k], np.dot(np.diag(S[:k]), VT[:k, :]))
    plt.subplot(2, len(k_values) + 1, i + 2)
    plt.imshow(compressed, cmap='gray')
    plt.title(f'Good\nk={k} ({pct:.1f}%)')
    plt.axis('off')

    # Bad compression (e.g., flatten S or scramble)
    bad_S = np.random.permutation(S[:k])  # randomize top-k singular values
    bad_compressed = np.dot(U[:, :k], np.dot(np.diag(bad_S), VT[:k, :]))
    plt.subplot(2, len(k_values) + 1, len(k_values) + i + 3)
    plt.imshow(bad_compressed, cmap='gray')
    plt.title(f'Bad\nk={k} ({pct:.1f}%)')
    plt.axis('off')
"""

# Console Full Print-out
print(f"Original number of singular values (k max): {k_max}")
for k in k_values:
    pct = (k / k_max) * 100
    print(f"k = {k} → Retains {pct:.2f}% of the image data")

# Shows the images in one window
plt.tight_layout()
plt.show()

# Save Comrpessed Image
# cv2.imwrite('compressed_flower.jpg', compressed)


