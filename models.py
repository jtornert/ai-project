import tensorflow as tf

checkpoint_path_mean = 'training_mean/cp.ckpt'
checkpoint_path_deep = 'training_deep/cp.ckpt'
checkpoint_path_max = 'training_max/cp.ckpt'

model_type = 'None'


def mean(load=False):
    """
    A network with a hidden layer that consists of a number of nodes that is the mean value of the number of inputs and outputs.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=62, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    global model_type
    model_type = 'mean'

    if load == True:
        model.load_weights(checkpoint_path_mean)

    return model


def max(load=False):
    """
    A network with as many nodes in the hidden layer as the input layer.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=784, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=62, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    global model_type
    model_type = 'max'

    if load == True:
        model.load_weights(checkpoint_path_max)

    return model


def deep(load=False):
    """
    A deep learning network with three hidden layers.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=62, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    global model_type
    model_type = 'deep'

    if load == True:
        model.load_weights(checkpoint_path_deep)

    return model


def train(model, samples, labels, epochs=1, save=False):
    global checkpoint_path
    if model_type == 'mean':
        checkpoint_path = checkpoint_path_mean
    elif model_type == 'deep':
        checkpoint_path = checkpoint_path_deep
    elif model_type == 'max':
        checkpoint_path = checkpoint_path_max
    else:
        raise ValueError('Unknown type.')
    model.fit(samples, labels, epochs=epochs, callbacks=(
        None if save == False else [tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True, verbose=1)]))
