import cv2
import numpy as np
from collections import defaultdict
from time import time
from tqdm import tqdm

image_input = cv2.imread("test_data/color_mask.png")
mask_input = cv2.imread("test_data/mask.png")

# Measure execution time
start_time = time()

# Get inverse polygons mask
_, mask_inverse = cv2.threshold(
    cv2.cvtColor(mask_input, cv2.COLOR_BGR2GRAY), 0, 1, cv2.THRESH_BINARY
)

# Get connected components
num_polygons, polygons_connected_components = cv2.connectedComponents(mask_inverse)


# Cache the indices of each region to speed up points access
# Separate cache provided for faster access by index using `image[x, y]`
polygons_cache_x = defaultdict(list)
polygons_cache_y = defaultdict(list)

print("Cache polygons")
for i in tqdm(range(0, polygons_connected_components.shape[0])):
    for j in range(0, polygons_connected_components.shape[1]):
        val = polygons_connected_components[i, j]
        polygons_cache_x[val].append(i)
        polygons_cache_y[val].append(j)


image_filled_polygons = np.zeros_like(mask_input)

print("Fill polygons")
for label in tqdm(range(1, num_polygons)):

    x = polygons_cache_x[label]
    y = polygons_cache_y[label]

    region_pixels = image_input[x, y]

    # We can take only half of the pixels
    colors, counts = np.unique(
        region_pixels,
        return_counts=True,
        axis=0,
    )
    most_common_color = colors[np.argmax(counts)]

    # Fill the region with the most frequent color in the result image
    image_filled_polygons[x, y] = most_common_color

print(f"Execution time: {time() - start_time:.2f} s.")

# Save the result image
cv2.imwrite("test_data/mask_filled_opencv.png", image_filled_polygons)
