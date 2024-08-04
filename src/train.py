import tensorflow as tf

def create_model(layers, learning_rate=0.001, activation=None):
    try:
        tf.keras.utils.set_random_seed(42)
        model = tf.keras.Sequential()

        for layer in layers[:-1]:
            if activation:
                model.add(tf.keras.layers.Dense(layer, activation=activation))
            else:
                model.add(tf.keras.layers.Dense(layer))
        
        model.add(tf.keras.layers.Dense(layers[-1], activation='sigmoid' if layers[-1] == 1 else None))

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            metrics=['accuracy']
        )

        return model
    except Exception as e:
        print(f"Error in create_model: {e}")
        return None

def train_model(model, x_train, y_train, epochs=50, callbacks=None, verbose=0):
    try:
        history = model.fit(x_train, y_train, epochs=epochs, verbose=verbose, callbacks=callbacks)
        return history
    except Exception as e:
        print(f"Error in train_model: {e}")
        return None

def evaluate_model(model, x_train, y_train):
    try:
        return model.evaluate(x_train, y_train)
    except Exception as e:
        print(f"Error in evaluate_model: {e}")
        return None
