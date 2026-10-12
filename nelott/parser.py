import struct
import gzip
import numpy as np


# WHY: MNIST files are raw binary files (not text, not CSV).
# They use a format called IDX. Every IDX file starts with a small
# "header" that tells you: how many images, how tall, how wide.
# After the header comes the actual pixel data — just bytes 0-255.
#
# Header layout (each number is 4 bytes, big-endian):
#   [magic number]  [count]  [rows]  [cols]
#       2051         60000     28      28
#
# Labels files are simpler — no rows/cols, just count then one byte per label.


def read_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        # Read the 4-number header
        magic, count, rows, cols = struct.unpack('>IIII', f.read(16))

        # Sanity check: MNIST image files always have magic number 2051
        assert magic == 2051, "This doesn't look like an MNIST images file"

        # Read all remaining bytes — these are the pixels
        raw = f.read()

    # Turn the flat list of bytes into a 3D array: (count, rows, cols)
    images = np.frombuffer(raw, dtype=np.uint8)
    images = images.reshape(count, rows, cols)

    return images   # shape: (60000, 28, 28), values: 0-255


def read_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        # Labels header is only 2 numbers: magic + count (count not needed, numpy infers it)
        magic, _ = struct.unpack('>II', f.read(8))

        assert magic == 2049, "This doesn't look like an MNIST labels file"

        raw = f.read()

    labels = np.frombuffer(raw, dtype=np.uint8)  # shape: (60000,), values: 0-9

    return labels


def load_mnist(data_dir):
    x_train = read_images(f"{data_dir}/train-images-idx3-ubyte.gz")
    y_train = read_labels(f"{data_dir}/train-labels-idx1-ubyte.gz")
    x_test  = read_images(f"{data_dir}/t10k-images-idx3-ubyte.gz")
    y_test  = read_labels(f"{data_dir}/t10k-labels-idx1-ubyte.gz")

    return x_train, y_train, x_test, y_test
