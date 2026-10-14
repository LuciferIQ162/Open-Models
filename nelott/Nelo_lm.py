from parser import load_mnist

x_train, y_train, x_test, y_test = load_mnist('datasets')

# Normalize: pixels go from 0-255 down to 0.0-1.0
x_train = x_train / 255.0
x_test  = x_test  / 255.0


# --- Retrieving a single image and its label ---

i = 0                    # pick any number 0 to 59999
image = x_train[i]       # one 28x28 grid of pixel values
label = y_train[i]       # the digit it shows (0-9)

print("Label:", label)          # e.g. 5
print("Image shape:", image.shape)  # (28, 28)
print("Top-left pixel:", image[0][0])  # a value between 0.0 and 1.0


# --- Retrieving a batch (a group of images at once) ---

batch_images = x_train[0:32]   # first 32 images,  shape: (32, 28, 28)
batch_labels = y_train[0:32]   # their labels,      shape: (32,)

print("Batch size:", len(batch_images))   # 32
print("Labels in batch:", batch_labels)   # e.g. [5 0 4 1 9 2 1 ...]


# --- Looking up all images of a specific digit ---

fives = x_train[y_train == 5]   # every image where the label is 5
print("Number of 5s in training set:", len(fives))  # ~6000
