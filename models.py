import tensorflow as tf

checkpoint_path_mean = 'training_mean/cp.ckpt'
checkpoint_path_deep = 'training_deep/cp.ckpt'
checkpoint_path_max = 'training_max/cp.ckpt'

optimizer = 'adam'
loss = tf.keras.losses.sparse_categorical_crossentropy
# tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

dropout = 0.2


def mean(load=False):
    """
    A network with a hidden layer that consists of a number of nodes that is the mean value of the number of inputs and outputs.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(units=62, activation=tf.nn.softmax))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    if load == True:
        model.load_weights(checkpoint_path_mean)

    return model


def max(load=False):
    """
    A network with as many nodes in the hidden layer as the input layer.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(units=784, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(units=62, activation=tf.nn.softmax))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    if load == True:
        model.load_weights(checkpoint_path_max)

    return model


def deep(load=False):
    """
    A deep learning network with three hidden layers.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(units=62, activation=tf.nn.softmax))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    if load == True:
        model.load_weights(checkpoint_path_deep)

    return model


def train(model, samples, labels, epochs=1, save=False, cppath=None):
    model.fit(samples, labels, epochs=epochs, callbacks=(
        None if save == False else [tf.keras.callbacks.ModelCheckpoint(
            filepath=cppath, save_weights_only=True, verbose=1)]))
