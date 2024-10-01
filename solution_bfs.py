import cv2
import numpy as np
from tqdm import tqdm
from time import time


image_input = cv2.imread("test_data/color_mask.png")
mask_input = cv2.imread("test_data/mask.png")


def extract_chunk_from_point(point: np.ndarray):
    """
    BFS algorithm to extract chunk of points
    :param point: starting point
    :return: extracted chunk colors and points
    """
    current_color = image_input[point[0], point[1]]
    extracted_chunk_points = [point]
    extracted_chunk_colors = [current_color]

    points_queue = [point]

    while points_queue:
        point = points_queue.pop(0)

        for new_point in [
            [point[0] - 1, point[1]],
            [point[0] + 1, point[1]],
            [point[0], point[1] - 1],
            [point[0], point[1] + 1],
        ]:
            if (
                new_point[0] < 0
                or new_point[0] >= image_input.shape[0]
                or new_point[1] < 0
                or new_point[1] >= image_input.shape[1]
            ):
                continue

            if visited_points[new_point[0], new_point[1]] == 1:
                continue

            visited_points[new_point[0], new_point[1]] = 1

            extracted_chunk_colors.append(image_input[new_point[0], new_point[1]])
            points_queue.append(new_point)
            extracted_chunk_points.append(new_point)

    return np.array(extracted_chunk_colors), np.array(extracted_chunk_points)


# Measure execution time
start_time = time()


image_filled_polygons = np.zeros_like(image_input)

# Get coordinates of all points
all_points_coordinates = np.array(
    np.meshgrid(range(mask_input.shape[0]), range(mask_input.shape[1]))
).T.reshape(-1, 2)

# Create mask of polygons. This should be visited points in BFS algorith from the start
_, visited_points = cv2.threshold(
    cv2.cvtColor(mask_input, cv2.COLOR_BGR2GRAY), 0, 1, cv2.THRESH_BINARY_INV
)

points_counter = 0

t = tqdm()
print("Fill polygons")
while True:
    t.update(1)

    # Select next unvisited point
    while points_counter < len(all_points_coordinates):
        current_point = all_points_coordinates[points_counter]
        points_counter += 1
        # Meet the first unchecked point
        if visited_points[current_point[0], current_point[1]] == 0:
            break

    # Finish when all points are visited
    if points_counter >= len(all_points_coordinates):
        break

    chunk_colors, chunk_points = extract_chunk_from_point(current_point)

    # Select the most frequent color in the region
    colors, counts = np.unique(chunk_colors, return_counts=True, axis=0)
    most_common_color = colors[np.argmax(counts)]

    # Fill the region with the most frequent color in the result image
    image_filled_polygons[chunk_points[:, 0], chunk_points[:, 1]] = most_common_color


print(f"Execution time: {time() - start_time:.2f} s.")

# Save the result image
cv2.imwrite("test_data/mask_filled_bfs.png", image_filled_polygons)
