import tensorflow as tf

checkpoint_path_dual = 'training_dual/cp.ckpt'
checkpoint_path_letters = 'training_letters/cp.ckpt'

model_type = 'None'


def dual(load=False):
    """
    The simplest available network that solves the problem.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=36, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    global model_type
    model_type = 'dual'

    if load == True:
        model.load_weights(checkpoint_path_dual)

    return model


def digits(load=False):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def letters(load=False):
    """
    Neural network model for learning the set of letters in the EMNIST dataset.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=26, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    global model_type
    model_type = 'letters'

    if load == True:
        model.load_weights(checkpoint_path_letters)

    return model


def train(model, samples, labels, epochs=1, save=False):
    global checkpoint_path
    if model_type == 'dual':
        checkpoint_path = checkpoint_path_dual
    elif model_type == 'letters':
        checkpoint_path = checkpoint_path_letters
    else:
        raise ValueError('Unknown type.')
    model.fit(samples, labels, epochs=epochs, callbacks=(
        None if save == False else [tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True, verbose=1)]))
