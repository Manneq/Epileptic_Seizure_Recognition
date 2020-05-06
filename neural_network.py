import keras


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
                             min_delta=1e-4,
                             patience=5,
                             restore_best_weights=True),
                         keras.callbacks.ReduceLROnPlateau(),
                         keras.callbacks.TensorBoard(
                             log_dir="data/output/neural_network/logs",
                             batch_size=batch_size,
                             write_grads=True,
                             write_images=True)])

    return model


def model_evaluation(model, validation_set,
                     batch_size=512):
    print("Metrics results: ", model.metrics_names)
    print(model.evaluate(x=validation_set[0],
                         y=validation_set[1],
                         batch_size=batch_size))

    model.save_weights("data/output/neural_network/weights/weights.h5")

    return


def model_creation_and_compiling(input_dimension):
    model = keras.Sequential()

    model.add(keras.layers.Dense(1024, input_dim=input_dimension,
                                 activation='sigmoid'))
    model.add(keras.layers.Dense(512, activation='sigmoid'))
    model.add(keras.layers.Dense(256, activation='sigmoid'))
    model.add(keras.layers.Dense(128, activation='sigmoid'))
    model.add(keras.layers.Dense(64, activation='sigmoid'))
    model.add(keras.layers.Dense(32, activation='sigmoid'))
    model.add(keras.layers.Dense(16, activation='sigmoid'))
    model.add(keras.layers.Dense(8, activation='sigmoid'))
    model.add(keras.layers.Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model


def neural_network_model(training_set, validation_set,
                         training=True):
    model = model_creation_and_compiling(training_set[0].shape[0])

    if training:
        model = model_training(model, training_set, validation_set)
        model_evaluation(model, validation_set)
    else:
        model.load_weights("data/output/neural_network/weights/weights.h5")
        model_evaluation(model, validation_set)

    return
