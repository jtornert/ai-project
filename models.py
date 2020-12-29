import tensorflow as tf

checkpoint_path_dual = 'training_dual/cp.ckpt'
checkpoint_path_deep = 'training_deep/cp.ckpt'

model_type = 'None'


def dual(load=False):
    """
    A single network that deals with both digits and letters.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=36, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    global model_type
    model_type = 'dual'

    if load == True:
        model.load_weights(checkpoint_path_dual)

    return model


def deep(load=False):
    """
    A single network that deals with both digits and letters.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=410, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=36, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    global model_type
    model_type = 'deep'

    if load == True:
        model.load_weights(checkpoint_path_dual)

    return model


def train(model, samples, labels, epochs=1, save=False):
    global checkpoint_path
    if model_type == 'dual':
        checkpoint_path = checkpoint_path_dual
    elif model_type == 'deep':
        checkpoint_path = checkpoint_path_deep
    else:
        raise ValueError('Unknown type.')
    model.fit(samples, labels, epochs=epochs, callbacks=(
        None if save == False else [tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True, verbose=1)]))
