import numpy as np
import neurolab as nl
import numpy.random as rand
import pylab as pl

skv = 0.05
center = np.array([[0.2, 0.2], [0.4, 0.4], [0.7, 0.3], [0.2, 0.5]])
rand_norm = skv * rand.randn(100, 4, 2)
inp = np.array([center + r for r in rand_norm])
inp.shape = (100 * 4, 2)
rand.shuffle(inp)
net = nl.net.newc([[0.0, 1.0], [0.0, 1.0]], 4)
error = net.train(inp, epochs=200, show=20)
fig, axs = pl.subplots(2)
fig.suptitle('Classification Problem')
axs.flat[1].set(xlabel='Epoch number', ylabel='error (default MAE)')
axs[1].plot(error)
w = net.layers[0].np['w']
axs[0].plot(inp[:, 0], inp[:, 1], '.', center[:, 0], center[:, 1], 'yv', w[:, 0],
            w[:, 1], 'p')
axs[0].legend(['train samples', 'real centers', 'train centers'], loc='upper left')
pl.show()
