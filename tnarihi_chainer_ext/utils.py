import numpy as np
import six
from concurrent.futures import ThreadPoolExecutor


def data_provider(num_examples, batch_size, load_func, rng=None):
    """
    Generator function which yields tuple of blobs infinitely as long as you
    request.

    Each generator call creates threads to pre-fetch next blobs in
    background before yielding blobs.

    Args:

        num_examples (int): Number of examples of your dataset. Random sequence
            of indexes is generated according to this number.
        batch_size (int): Number of examples in a batch to be loaded for
            yielding blobs
        load_func (function): Takes a single argument `i`, an index of an
            example in your dataset to be loaded, and returns a tuple of data.
            Every calls by any index `i` must returns a tuple of arrays with
            the same shape.
        rng: Numpy random number generator

    Here is an example of `load_func`. This can be used for a common
    classification dataset.

    .. code-block:: python
        import numpy as np
        from scipy.misc import imread

        image_paths = load_image_paths()
        labels = load_labels()

        def my_load_func(i):
            '''
            Returns:
                image: c x h x w array
                label: 0-shape array
            '''
            img = imread(image_paths[i]).astype('float32')
            return np.rollaxis(img, 2), np.array(labels[i])
    """
    if rng is None:
        rng = np.random.RandomState(313)

    def threads_generator():
        threads = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                inds = rng.permutation(num_examples)
                for i in inds:
                    threads += [executor.submit(load_func, i)]
                    if len(threads) == batch_size:
                        yield threads
                        threads = []
    # load to determine the blob shape
    shapes = [datum.shape for datum in load_func(0)]
    tgen = threads_generator()
    threads = tgen.next()
    while True:
        blobs = [np.zeros((batch_size,) + shape, dtype='float32')
                 for shape in shapes]
        for j, t in enumerate(threads):
            for datum, blob in six.moves.zip(t.result(), blobs):
                blob[j] = datum
        threads = tgen.next()
        yield tuple(blobs)


def blob_to_tile(blob, padsize=1, padval=0, normalize=True):
    """
    take an array of shape (n, channels, height, width)
    and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """
    assert(blob.ndim == 4)
    if blob.shape[1] != 3:
        blob = blob.reshape((-1, 1) + blob.shape[2:])
    blob = blob.transpose(0, 2, 3, 1)

    if normalize:
        blob = blob - blob.min()  # copy
        blob /= blob.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(blob.shape[0])))
    padding = (
        (0, n ** 2 - blob.shape[0]),
        (0, padsize),
        (0, padsize)
    ) + ((0, 0),) * (blob.ndim - 3)
    blob = np.pad(
        blob, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    blob = blob.reshape(
        (n, n)
        + blob.shape[1:]
    ).transpose((0, 2, 1, 3) + tuple(range(4, blob.ndim + 1)))
    blob = blob.reshape(
        (n * blob.shape[1], n * blob.shape[3]) + blob.shape[4:])
    if blob.shape[2] == 1:
        return blob.reshape((blob.shape[:2]))
    return blob
