import keras
import numpy as np


def model_training(model, training_set, validation_set,
                   batch_size=512, epochs=100):
    model.fit(x=training_set[0],
              y=training_set[1],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(validation_set[0],
                               validation_set[1]),
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

    model.save_weights("output_data/multivariate_analysis/initial/"
                       "neural_network/weights/weights.h5")

    return model


def model_evaluation(model, validation_set,
                     batch_size=512):
    metrics = model.evaluate(x=validation_set[0],
                             y=validation_set[1],
                             batch_size=batch_size)

    return metrics[1]


def model_creation_and_compiling():
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

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model


def neural_network_model(training_set, validation_set,
                         training=True):
    model = model_creation_and_compiling()

    training_set = (training_set[0],
                    training_set[1].drop(columns=['categories'], axis=1))
    validation_set_temp = (validation_set[0],
                           validation_set[1].drop(columns=['categories'],
                                                  axis=1))

    if training:
        model = model_training(model, training_set, validation_set_temp)
        model_evaluation(model, validation_set_temp)
    else:
        model.load_weights("output_data/multivariate_analysis/initial/"
                           "neural_network/weights/weights.h5")
        model_evaluation(model, validation_set_temp)

    return
