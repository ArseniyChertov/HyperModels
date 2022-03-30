import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Conv2D


class config():
    epochs = 10
    batch_size = 1024
    workers = 1
    optimizer = "adam"
    loss = "categorical_crossentropy"


class mnist_cnn():
    def initialize(hp):
        input = Input(shape=(28, 28, 1))
        first_dropout = hp.Choice('first_dropout', [0.0, 0.2, 0.4])
        x = input
        x = Dropout(first_dropout)(x)

        network_depth = hp.Choice('network_depth', [1, 2, 3, 4])
        init_n_flters = hp.Choice('n_filters', [4, 8, 16])
        n_filters_growth_factor = hp.Choice('n_filters_growth_factor', [2, 4, 8])


        for i in range(0, network_depth):
            depth = (n_filters_growth_factor ** i) * init_n_flters
            x = Conv2D(depth, 3, strides=(1, 1), activation='relu', padding='valid')(x)
            conv_dropout_k = hp.Choice('conv_dropout_' + str(i), [0.0, 0.25, 0.5])
            x = Dropout(conv_dropout_k)(x)

        x = Flatten()(x)

        units_dense = hp.Choice('units_dense', [4, 8, 16, 32])
        x = Dense(units_dense)(x)

        second_dropout = hp.Choice('second_dropout', [0.0, 0.2, 0.4, 0.6])
        x = Dropout(second_dropout)(x)

        x = Dense(10, activation="softmax")(x)

        model = Model(inputs=input, outputs=x)

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=config.loss,
                      metrics=['accuracy'])

        return model

