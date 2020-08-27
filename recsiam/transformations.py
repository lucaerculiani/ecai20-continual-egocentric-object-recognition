
import numpy as np
import skimage.draw

class RandomShapeOcclusion(object):

    def __init__(self, seed, prob):
        self.seed = seed
        self.prob = prob

        self.rnd = np.random.RandomState(seed)

    def transform(self, sequence):

        if self.rnd.uniform() >= self.prob:
            return sequence
        else:
            sequence = sequence.copy()

        shape = sequence.shape[1:3]

        image, labels = skimage.draw.random_shapes(shape, 1,
                                                   intensity_range=((127,127),),
                                                   num_trials=100000,
                                                   random_seed=self.rnd.randint(2**31))


        target = image == 127

        ranges = self.rnd.randint(sequence.shape[0], size=2)
        ranges = np.sort(ranges)

        
        sequence[ranges[0]:ranges[1], target] = 127

        return sequence








