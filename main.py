import emnist
import models
import tests

images, labels = emnist.extract_training_samples('byclass')

model_mean = models.mean(load=True)
model_max = models.max(load=True)
model_deep = models.deep(load=True)

print('\nTraining model: mean')
models.train(model_mean, images, labels, epochs=20,
             save=True, cppath=models.checkpoint_path_mean)
print('\nTraining model: max')
models.train(model_max, images, labels, epochs=20,
             save=True, cppath=models.checkpoint_path_max)
print('\nTraining model: deep')
models.train(model_deep, images, labels, epochs=20,
             save=True, cppath=models.checkpoint_path_deep)

# print('\nTesting model: mean')
# tests.rand(model_mean, iter=5)
# tests.paint(model_mean)
# tests.nist(model_mean)

# print('\nTesting model: max')
# tests.rand(model_max, iter=5)
# tests.paint(model_max)
# tests.nist(model_max)

# print('\nTesting model: deep')
# tests.rand(model_deep, iter=5)
# tests.paint(model_deep)
# tests.nist(model_deep)

# print('\nDone')
