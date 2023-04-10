# import numpy as np
# import matplotlib.pyplot as plt
# # center_new = (27, 12)
# def create_circle_array(shape, center, radius):
#     x, y = np.ogrid[:shape[0], :shape[1]]
#     x, y = x - center[0], y - center[1]
#     mask = x**2 + y**2 <= radius**2
#     return np.where(mask, 1, 0)

# shape = (30, 60)
# center = (13, 27)
# radius = 10

# circle_array = create_circle_array(shape, center, radius)

# # fiducial_shape = ()
# # circle_array = create_circle_array()

# plt.figure()
# plt.imshow(circle_array)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def create_circle(rows, cols, center, radius):
    """
    Function to create a filled circle of ones in a numpy array of zeros
    """
    image = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                image[i, j] = 1
    return image

# Define the size of the array
rows, cols = 30, 60

# Create the array of zeros
image = np.zeros((rows, cols), dtype=np.uint8)

# Define the center coordinates of the two circles
center1 = (13, 27)
center2 = (13, 4.5)

# Define the radius of the two circles
radius1 = 10
radius2 = 2.5

# Create the first circle
circle1 = create_circle(rows, cols, center1, radius1)

# Create the second circle
circle2 = create_circle(rows, cols, center2, radius2)

# Add the two circles to the array of zeros
image = np.add(image, circle1)
image = np.add(image, circle2)

image[image==1] = 2
image[image==0] = 1
image[image==2] = 0

# Display the image
plt.imshow(image)
plt.show()
