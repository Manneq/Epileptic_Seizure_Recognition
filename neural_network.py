"""
    File 'neural_network.py' used for creating, training and evaluating
        neural network.
"""
import keras


def model_training(model, training_set, validation_set,
                   batch_size=512, epochs=100):
    """
        Method for neural network training.
        param:
            1. model - Keras neural network model
            2. training_set - tuple of sets for training
            3. validation_set - tuple of sets for validation
            4. batch_size - batch size for training (512 as default)
            5. epochs - number of epochs (100 as default)
        return:
            model - Keras neural network model
    """
    model.fit(x=training_set[0],
              y=training_set[1].drop(columns=['categories'], axis=1),
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(validation_set[0],
                               validation_set[1].drop(columns=['categories'],
                                                      axis=1)),
              shuffle=True,
              callbacks=[keras.callbacks.TerminateOnNaN(),
                         keras.callbacks.EarlyStopping(
                             monitor='val_acc',
                             min_delta=1e-5,
                             patience=5,
                             restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(),
                         keras.callbacks.TensorBoard(
                             log_dir="output_data/multivariate_analysis/"
                                     "initial/neural_network/logs",
                             batch_size=batch_size,
                             write_grads=True,
                             write_images=True)])

    # Weights saving
    model.save_weights("output_data/multivariate_analysis/initial/"
                       "neural_network/weights/weights.h5")

    return model


def model_evaluation(model, validation_set,
                     batch_size=512):
    """
        Method to evaluate model on validation dataset.
        param:
            model - Keras neural network model
            validation_set - tuple sets for validation
            batch_size - batch size for evaluation (512 as default)
        return:
            accuracy
    """
    metrics = model.evaluate(x=validation_set[0],
                             y=validation_set[1].drop(columns=['categories'],
                                                      axis=1),
                             batch_size=batch_size)

    return metrics[1]


def model_creation_and_compiling():
    """
        Method for neural network creation.
        return:
            model - Keras neural network model
    """
    # Sequential model using
    model = keras.Sequential()

    model.add(keras.layers.Dense(4096, input_dim=178,
                                 activation='relu'))
    model.add(keras.layers.Dense(2048, activation='relu'))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(5, activation='softmax'))

    # Model compiling
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model


def neural_network_model(training_set, validation_set,
                         training=True):
    """
        Method for neural network using for classification.
        param:
            1. training_set - tuple of sets for training
            2. validation_set - tuple of sets for validation
            3. training - boolean value for training (True as default)
    """
    # Model creation
    model = model_creation_and_compiling()

    if training:
        # Model training and evaluation
        model = model_training(model, training_set, validation_set)
        model_evaluation(model, validation_set)
    else:
        # Weights loading and model evaluation
        model.load_weights("output_data/multivariate_analysis/initial/"
                           "neural_network/weights/weights.h5")
        model_evaluation(model, validation_set)

    return
