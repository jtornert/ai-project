import tensorflow as tf

checkpoint_path_simplest = 'training_simplest/cp.ckpt'

model_type = 'None'


def simplest(load=False):
    """
    The simplest available network that solves the problem.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    global model_type
    model_type = 'simplest'

    if load == True:
        model.load_weights(checkpoint_path_simplest)

    return model


def train(model, samples, labels, epochs=1, save=False):
    global checkpoint_path
    if model_type == 'simplest':
        checkpoint_path = checkpoint_path_simplest
    else:
        raise ValueError('Unknown type.')
    model.fit(samples, labels, epochs=epochs, callbacks=(
        None if save == False else [tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path_simplest, save_weights_only=True, verbose=1)]))
