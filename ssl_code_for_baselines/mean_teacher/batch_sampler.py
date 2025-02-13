import itertools
import os.path
from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler


class TransformTwice:
    """
    Apply two transforms to the same input
    """
    def __init__(self, transform, noise_transform):
        self.transform = transform
        self.noise_transform = noise_transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.noise_transform(inp)
        return out1, out2


class TwoStreamBatchSampler(Sampler):
    def __init__(self, labeled_idxs, unlabeled_idxs, batch_size, labeled_batch_size):
        """
        Custom batch sampler for labeled and unlabeled data.

        Args:
            labeled_idxs (list[int]): Indices for labeled data.
            unlabeled_idxs (list[int]): Indices for unlabeled data.
            batch_size (int): Total batch size.
            labeled_batch_size (int): Number of labeled samples per batch.
        """
        self.labeled_idxs = labeled_idxs
        self.unlabeled_idxs = unlabeled_idxs
        self.batch_size = batch_size
        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = batch_size - labeled_batch_size

        assert len(labeled_idxs) >= labeled_batch_size, \
            "Not enough labeled samples to satisfy batch size."
        assert len(unlabeled_idxs) >= self.unlabeled_batch_size, \
            "Not enough unlabeled samples to satisfy batch size."

    def __iter__(self):
        # Shuffle indices
        labeled_perm = np.random.permutation(self.labeled_idxs)
        unlabeled_perm = np.random.permutation(self.unlabeled_idxs)

        # Create batches
        labeled_batches = [
            labeled_perm[i:i + self.labeled_batch_size]
            for i in range(0, len(labeled_perm), self.labeled_batch_size)
        ]
        unlabeled_batches = [
            unlabeled_perm[i:i + self.unlabeled_batch_size]
            for i in range(0, len(unlabeled_perm), self.unlabeled_batch_size)
        ]

        # Combine batches
        for labeled_batch, unlabeled_batch in zip(labeled_batches, unlabeled_batches):
            yield np.concatenate((labeled_batch, unlabeled_batch))

    def __len__(self):
        return len(self.labeled_idxs) // self.labeled_batch_size

    
class SingleStreamBaselineSampler(Sampler):
    """
    Iterate over a single set of values

    An 'epoch' is one iteration through the primary indices.
    This is for baseline computation with a subset of only labeld data.
    """
    def __init__(self, primary_indices, batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        return (
            primary_batch 
            for primary_batch
            in  grouper(primary_iter, self.primary_batch_size)
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)
