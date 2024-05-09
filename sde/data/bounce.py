import numpy as np


def get_template(size):
    # draw a anti aliased circle
    grid = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    d = np.sqrt((grid ** 2).sum(0))
    template = 1. / (1. + np.exp(20 * (d - 1.)))
    return template.clip(0, 1).astype(np.float32)


class Bounce:
    def __init__(self, train, sequence_length, size=64, dt=.1):
        self.train = train
        self.sequence_length = sequence_length  
        self.size = size 
        self.bsize = size // 4
        self.dt = dt
        self.template = get_template(self.bsize)

    def __len__(self):
        if self.train:
            return 6000
        else:
            return 300

    def __getitem__(self, index):
        if not self.train:
            np.random.seed(index)

        x = np.zeros((self.sequence_length, self.size, self.size, 1), dtype=np.float32)

        sx = np.random.randint(self.size - self.bsize)
        sy = np.random.randint(self.size - self.bsize)
        dx = np.random.randint(-4, 5)
        dy = np.random.randint(-4, 5)
        for t in range(self.sequence_length):
            if sy < 0:
                sy = 0 
                dy = np.random.randint(1, 5)
                dx = np.random.randint(-4, 5)
            elif sy >= self.size - self.bsize:
                sy = self.size - self.bsize - 1
                dy = np.random.randint(-4, 0)
                dx = np.random.randint(-4, 5)
            if sx < 0:
                sx = 0
                dx = np.random.randint(1, 5)
                dy = np.random.randint(-4, 5)
            elif sx >= self.size - self.bsize:
                sx = self.size - self.bsize - 1
                dx = np.random.randint(-4, 0)
                dy = np.random.randint(-4, 5)

            x[t, sy:sy+self.bsize, sx:sx+self.bsize, 0] = self.template
            sy += dy
            sx += dx
        return x
