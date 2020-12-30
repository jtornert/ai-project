import emnist
import models
import tests

images, labels = emnist.extract_training_samples('byclass')

model_mean = models.mean(load=False)
model_max = models.max(load=False)
model_deep = models.deep(load=False)

models.train(model_mean, images, labels, epochs=1, save=True)
models.train(model_max, images, labels, epochs=1, save=True)
models.train(model_deep, images, labels, epochs=1, save=True)

tests.rand(model_mean, iter=5)
tests.paint(model_mean)
tests.nist(model_mean)

tests.rand(model_max, iter=5)
tests.paint(model_max)
tests.nist(model_max)

tests.rand(model_deep, iter=5)
tests.paint(model_deep)
tests.nist(model_deep)
