from PIL import Image
import numpy as np
from utils.minisom import MiniSom
import matplotlib.pyplot as plt

img = plt.imread('test.jpg')
pixels = np.resize(img, (int(img.shape[0]/ 20), int(img.shape[1]/20), 3))
pixels = np.reshape(pixels, (pixels.shape[0]*pixels.shape[1], 3)) / 255.

som = MiniSom(x=14, y=25, input_len=3, sigma=1, learning_rate=0.2, neighborhood_function='bubble')
som.random_weights_init(pixels)
starting_weights = som.get_weights().copy()

som.train(pixels, 10000, random_order=True, verbose=True)
qnt = som.quantization(pixels)
clustered = np.reshape(qnt, (int(img.shape[0]/ 20), int(img.shape[1]/20), 3))
# clustered = np.zeros(img.shape)
# for i, q in enumerate(qnt):  # place the quantized values into a new image
#     clustered[np.unravel_index(i, shape=(img.shape[0], img.shape[1]))] = q

plt.figure(figsize=(7, 7))
plt.figure(1)
plt.subplot(221)
plt.title('original')
plt.imshow(img)
plt.subplot(222)
plt.title('result')
plt.imshow(clustered)

plt.subplot(223)
plt.title('initial colors')
plt.imshow(starting_weights, interpolation='none')
plt.subplot(224)
plt.title('learned colors')
plt.imshow(som.get_weights(), interpolation='none')

plt.tight_layout()

plt.show()



