# Assumptions

- The solution is exact.
- No image rescaling
    - This will work with current problem but not robust to different image sizes and mask
      configurations
- No multi-threading used
    - Multi-threading solution can be implemented to speed up the process but not considered in my
      case since in general nothing will change significantly in the solution

# Approach #1

- Direct implementation of BFS algorithm similar to "Number of islands" problem
- Added this approach just to implement the direct solution for polygons extractions without
  `opencv.connectedComponents` function

To run it

```
pip install -r requirement.txt
python solution_bfs.py
```

Recorder execution time: `15.72 s.`

# Approach #2

- Use `cv2.connectedComponents` to find the connected components in the image (polygons).

To run it

```
pip install -r requirement.txt
python solution_opencv.py
```

Recorder execution time: `3.49 s.`

# Results

Filled mask for opencv solution -> `test_data/mask_filled_opencv.png`
Filled mask for BFS solution -> `test_data/mask_filled_bfs.png`