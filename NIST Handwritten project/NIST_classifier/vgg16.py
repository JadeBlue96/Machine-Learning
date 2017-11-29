

import numpy as np
import tensorflow as tf
import download
import os
from cache import cache
import sys
########################################################################

# https://github.com/pkmital/CADL/blob/master/session-4/libs/vgg16.py
# https://s3.amazonaws.com/cadl/models/synset.txt

# Internet URL for the file with the VGG16 model.
data_url = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"

# Directory to store the downloaded data.
data_dir = "vgg16/"

# File containing the TensorFlow graph definition. 
path_graph_def = "vgg16.tfmodel"

########################################################################


def maybe_download():

    print("Downloading VGG16 Model ...")
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


########################################################################


class VGG16:

    tensor_name_input_image = "images:0"
    tensor_name_transfer_layer = "pool5:0"
    layer_names = ['conv1_1/conv1_1', 'conv1_2/conv1_2',
                   'conv2_1/conv2_1', 'conv2_2/conv2_2',
                   'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
                   'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
                   'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']

    def __init__(self):

        self.graph = tf.Graph()

        with self.graph.as_default():

            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                # The graph-def is a saved copy of a TensorFlow graph.
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name='')

            # Get a reference to the tensor for inputting images to the graph.
            self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)

            # Get references to the tensors for the commonly used layers.
            self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]
            
            # Get the tensor for the last layer of the graph, aka. the transfer-layer.
            self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)

            # Get the number of elements in the transfer-layer.
            self.transfer_len = self.transfer_layer.get_shape()[1]

            # Create a TensorFlow session for executing the graph.
            self.session = tf.Session(graph=self.graph)

    def get_layer_tensors(self, layer_ids):
        """
        Return a list of references to the tensors for the layers with the given id's.
        """

        return [self.layer_tensors[idx] for idx in layer_ids]

    def get_layer_names(self, layer_ids):
        """
        Return a list of names for the layers with the given id's.
        """

        return [self.layer_names[idx] for idx in layer_ids]

    def get_all_layer_names(self, startswith=None):
        """
        Return a list of all the layers (operations) in the graph.
        The list can be filtered for names that start with the given string.
        """

        # Get a list of the names for all layers (operations) in the graph.
        names = [op.name for op in self.graph.get_operations()]

        if startswith is not None:
            names = [name for name in names if name.startswith(startswith)]

        return names

    def _create_feed_dict(self, image_path=None,image=None):
        """
        Create and return a feed-dict with an image.
        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).
        :return:
            Dict for feeding to the graph in TensorFlow.
        """

        # Expand 3-dim array to 4-dim by prepending an 'empty' dimension.        
        image = np.expand_dims(image, axis=0)
        if image is not None:
            feed_dict = {self.tensor_name_input_image: image}
        elif image_path is not None:
            # Read the image as an array of bytes.
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            feed_dict = {self.tensor_name_input_image: image_data}

        return feed_dict
    
    def transfer_values(self, image_path=None, image=None):
      

        # Create a feed-dict for the TensorFlow graph with the input image.
        feed_dict = self._create_feed_dict(image_path=image_path, image=image)

        transfer_values = self.session.run(self.transfer_layer, feed_dict=feed_dict)

        # Reduce to a 1-dim array.
        transfer_values = np.squeeze(transfer_values)

        return transfer_values
def process_images(fn, images=None, image_paths=None):
    """

    :param fn:
        Function to be called for each image.

    :param images:
        List of images to process.

    :param image_paths:
        List of file-paths for the images to process.

    :return:
        Numpy array with the results.
    """

    # using images or image_paths?
    using_images = images is not None

    # Number of images.
    if using_images:
        num_images = len(images)
    else:
        num_images = len(image_paths)

    # Pre-allocate list for the results.
    # This holds references to other arrays. Initially the references are None.
    result = [None] * num_images

    # For each input image.
    for i in range(num_images):
        # Status-message. 
        msg = "\r- Processing image: {0:>6} / {1}".format(i+1, num_images)

        # Print the status message.
        sys.stdout.write(msg)
        sys.stdout.flush()

        # Process the image and store the result for later use.
        if using_images:
            result[i] = fn(image=images[i])
        else:
            result[i] = fn(image_path=image_paths[i])

    # Print newline.
    print()

    # Convert the result to a numpy array.
    result = np.array(result)

    return result


########################################################################
def transfer_values_cache(cache_path, model, images=None, image_paths=None):


    # Helper-function for processing the images if the cache-file does not exist.
    def fn():
        return process_images(fn=model.transfer_values, images=images, image_paths=image_paths)

    # Read the transfer-values from a cache-file, or calculate them if the file does not exist.
    transfer_values = cache(cache_path=cache_path, fn=fn)

    return transfer_values

def transfer_values_calc(model, images=None, image_paths=None):


    # Helper-function for processing the images if the cache-file does not exist.
    def fn():
        return process_images(fn=model.transfer_values, images=images, image_paths=image_paths)

    # Read the transfer-values from a cache-file, or calculate them if the file does not exist.
    transfer_values = fn()

    return transfer_values





