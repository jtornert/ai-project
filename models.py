import tensorflow as tf

checkpoint_path_simplest = 'training_simplest/cp.ckpt'


def load_model(model, type):
    """
    Loads the weights from the checkpoint for the specified type of network.
    """
    print(f'Loading weights from model type: {type}')
    if type == 'simplest':
        model.load_weights(checkpoint_path_simplest)
    else:
        raise ValueError('Unknown type specified in modules.load_model!')


def simplest():
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
    return model


def train_simplest(model, samples, labels, epochs, save=False):
    if save == True:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path_simplest, save_weights_only=True, verbose=1)
        model.fit(samples, labels, epochs=epochs, callbacks=[cp_callback])
    else:
        model.fit(samples, labels, epochs=epochs)
