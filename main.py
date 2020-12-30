import emnist
import models
import tests

train_images, train_labels = emnist.extract_training_samples('byclass')

model = models.dual(load=True)

# models.train(model, train_images, train_labels, epochs=1, save=True)

# tests.rand(model)
# tests.paint(model)
tests.nist(model)
