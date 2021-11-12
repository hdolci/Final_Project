from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def main(rune):

    mpl.rcParams['figure.figsize'] = (8, 6)
    mpl.rcParams['axes.grid'] = False
    # get_ipython().run_line_magic('matplotlib', 'inline')

    df = pd.read_csv('./data/Rune_Data.csv')

    def univariate_data(dataset, start_index, end_index, history_size, target_size):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indices = range(i-history_size, i)
            # Reshape our data to (history_Size,1)
            data.append(np.reshape(dataset[indices], (history_size, 1)))
            labels.append(dataset[i+target_size])
        return np.array(data), np.array(labels)

    TRAIN_SPLIT = 120

    tf.random.set_seed(12)

    # Univariate Timeseries test Test 1: Soul_rune, user input

    # input options:
    runes = rune
    uni_data = df[runes]

    # uni_data.index = df['timestamp']

    uni_data = uni_data.values

    # normalize features by applying coefficent of variation. subtracting mean and dividing by standard dev
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()

    # normalized data, coefficent of variation
    uni_data = (uni_data-uni_train_mean)/uni_train_std

    # Univariate model. We feed the Model the last 10 records, based off the previous historical data. it needs to learn to predict the next upcoming record
    univariate_past_history = 10
    univariate_future_target = 0

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)

    # Create time step

    def create_time_steps(length):
        time_steps = []
        for i in range(-length, 0, 1):
            time_steps.append(i)
        return time_steps

    def show_plot(plot_data, delta, title):
        labels = ['History', 'True Future', 'Model Prediction']
        marker = ['.-', 'rx', 'go']
        time_steps = create_time_steps(plot_data[0].shape[0])
        if delta:
            future = delta
        else:
            future = 0

        plt.title(title)
        for i, x in enumerate(plot_data):
            if i:
                plt.plot(future, plot_data[i], marker[i], markersize=10,
                         label=labels[i])
            else:
                plt.plot(time_steps, plot_data[i].flatten(
                ), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future+5)*2])
        plt.xlabel('Time-Step')
        return plt

    # BASELINE

    def baseline(history):
        return np.mean(history)

    # Apply Recurring neural network to Rune's Dataset

    BATCH_SIZE = 20
    BUFFER_SIZE = 5

    train_univariate = tf.data.Dataset.from_tensor_slices(
        (x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    # Long short term memory Neural network, Tensorflow
    simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])

    simple_lstm_model.compile(optimizer='adam', loss='mae')

    for x, y in val_univariate.take(1):
        print(simple_lstm_model.predict(x).shape)

    EVALUATION_INTERVAL = 500
    EPOCHS = 20

    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, validation_steps=50)

    # Loss is decreasing, this is probably due to a case of overfitting.....

    # Predict with LSTM Model

    for x, y in val_univariate.take(3):
        plot = show_plot([x[0].numpy(), y[0].numpy(),
                          simple_lstm_model.predict(x)[0]], 0, 'Simple LTSM Model')
        plot.show()

    # Our model did not perform the best..... We should consider adding new features.

    add_features = ['Blood_rune', 'Law_rune', 'Nature_rune', 'Soul_rune']

    features = df[add_features]
    features.index = df['timestamp']

    features.plot(subplots=True)

    dataset = features.values

    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)

    #  coefficent of variation
    dataset = (dataset-data_mean/data_std)

    # now that our data is cv, lets try and predict 1 day ahead. with a multivariated data

    def multivariate_data(dataset, target, start_index, end_index, history_size,
                          target_size, step, single_step=False):
        data = []
        labels = []

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset) - target_size

        for i in range(start_index, end_index):
            indicies = range(i-history_size, i, step)
            data.append(dataset[indicies])

            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])

        return np.array(data), np.array(labels)

    # Split into training and test sets, shuffle and Batch repeat them once more. finally create a simple lstm model.
    past_history = 10
    future_target = 1

    STEP = 1

    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                       TRAIN_SPLIT, past_history,
                                                       future_target, STEP,
                                                       single_step=True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                                   TRAIN_SPLIT, None, past_history,
                                                   future_target, STEP,
                                                   single_step=True)

    print('Single window of past history : {}'.format(x_train_single[0].shape))

    train_data_single = tf.data.Dataset.from_tensor_slices(
        (x_train_single, y_train_single))
    train_data_single = train_data_single.cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_single = tf.data.Dataset.from_tensor_slices(
        (x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32,
                                               input_shape=x_train_single.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))

    single_step_model.compile(
        optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

    for x, y in val_data_single.take(1):
        print(single_step_model.predict(x).shape)

    single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                                steps_per_epoch=EVALUATION_INTERVAL,
                                                validation_data=val_data_single,
                                                validation_steps=50)

    def plot_train_history(history, title):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(loss))

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title(title)
        plt.legend()

        # plt.savefig(os.path.join(
        #     "templates/static/images", "0.png"))
        # plt.show()

    plot_train_history(single_step_history,
                       'single step training and validation loss')

    for x, y in val_data_single.take(3):
        plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                          single_step_model.predict(x)[0]], 1,
                         'single step prediction')
        # plt.savefig(os.path.join(
        #     "templates/static/images", x + ".png"))

        # plot.show()
