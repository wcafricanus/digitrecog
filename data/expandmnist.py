"""expand_mnist.py
~~~~~~~~~~~~~~~~~~

Take the 50,000 MNIST training images, and create an expanded set of
250,000 images, by displacing each training image up, down, left and
right, by one pixel.  Save the resulting file to
../data/mnist_expanded.pkl.gz.

Note that this program is memory intensive, and may not run on small
systems.

"""

from __future__ import print_function

#### Libraries

# Standard library
import pickle
import gzip
import os.path
import random

# Third-party libraries
import numpy as np

from imageprocess.elasticdistortion import elastic_transform

print("Expanding the MNIST training set")

dir_name = os.path.dirname(__file__)

if os.path.exists(dir_name+"/mnist_expanded.pkl.gz"):
    print("The expanded training set already exists.  Exiting.")
else:
    f = gzip.open(dir_name+"/mnist.pkl.gz", 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    expanded_training_pairs = []
    j = 0 # counter
    for x, y in zip(training_data[0], training_data[1]):
        expanded_training_pairs.append((x, y))
        image = np.reshape(x, (-1, 28))
        j += 1
        if j % 1000 == 0: print("Expanding image number", j)
        # iterate over data telling us the details of how to
        # do the displacement
        for i in range(4):
            new_img = elastic_transform(image, alpha=20, sigma=4)
            expanded_training_pairs.append((np.reshape(new_img, 784), y))
    random.shuffle(expanded_training_pairs)
    expanded_training_data = [list(d) for d in zip(*expanded_training_pairs)]
    print("Saving expanded data. This may take a few minutes.")
    f = gzip.open(dir_name+"/mnist_expanded.pkl.gz", "w")
    pickle.dump((expanded_training_data, validation_data, test_data), f)
    f.close()
