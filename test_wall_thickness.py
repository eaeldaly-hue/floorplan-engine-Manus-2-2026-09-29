import cv2
import numpy as np


mask = cv2.imread(
    "output/wall_regions.png",
    cv2.IMREAD_GRAYSCALE
)

if mask is None:
    raise FileNotFoundError(
        "wall_regions.png not found"
    )


binary = mask > 0

height, width = binary.shape


# Count white pixels around the image
white_pixels = np.sum(binary)
total_pixels = binary.size

print()
print("Wall Mask Analysis")
print("------------------")

print("Image size:", width, "x", height)

print(
    "White pixels:",
    int(white_pixels)
)

print(
    "White percentage:",
    round(
        white_pixels / total_pixels * 100,
        2
    ),
    "%"
)


# Horizontal thickness analysis
horizontal_thickness = []

for y in range(height):

    row = binary[y]

    count = np.sum(row)

    if count >= 100:

        horizontal_thickness.append(
            (y, int(count))
        )


# Vertical thickness analysis
vertical_thickness = []

for x in range(width):

    column = binary[:, x]

    count = np.sum(column)

    if count >= 100:

        vertical_thickness.append(
            (x, int(count))
        )


print()
print("Rows with significant wall evidence:")
print(len(horizontal_thickness))

print()
print("Columns with significant wall evidence:")
print(len(vertical_thickness))


print()
print("Largest horizontal evidence:")

for item in sorted(
    horizontal_thickness,
    key=lambda x: x[1],
    reverse=True
)[:20]:

    print(
        "row:",
        item[0],
        "pixels:",
        item[1]
    )


print()
print("Largest vertical evidence:")

for item in sorted(
    vertical_thickness,
    key=lambda x: x[1],
    reverse=True
)[:20]:

    print(
        "column:",
        item[0],
        "pixels:",
        item[1]
    )