# %
##submission python export from jupyter notebook

# %%
import sys
import tensorflow.keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# %%
print(f"Tensor Flow Version: {tf.__version__}")
print()
print(f"Python {sys.version}")
print("GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.get_logger().setLevel('ERROR')
tf.test.is_built_with_cuda()
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# %%
import pickle
dict = pickle.load(open('california-housing-dataset.pkl', 'rb'))
x_train, y_train = dict['x_train'], dict['y_train']
x_test, y_test = dict['x_test'], dict['y_test']




# %%
print(x_train.shape, y_train.shape)
print(x_train[1,:], y_train[1])

# %%
features = ['MedInc', 'HouseAge' ,'AveRooms' ,'AveBedrms' ,'Population', 'AveOccup' ,'Latitude' ,'Longitude' ]

def show_dataset_min_and_max(x, y, features):
    for i in range(len(features)):
        a = x[:,i]
        print('{}; [{}, {}]'.format(features[i], min(a), max(a),))
        print('Datatype: {}'.format(a.dtype))
    print('\nMax Price: {}; Min Price: {}'.format(max(y), min(y[:])))
    



# %%
show_dataset_min_and_max(x_train, y_train, features)

# %%
show_dataset_min_and_max(x_test, y_test, features)

# %%
print(len(y_train[np.where(y_train[np.where(y_train >= 4.8)] < 4.999)]))
print(len(y_train[y_train >=5]))
print(len(y_test[y_test >=5]))

# %%
indices_to_remove_train = np.where(y_train >= 5)[0]
indices_to_remove_test = np.where(y_test >= 5)[0]

x_train_no_5s = np.delete(x_train, indices_to_remove_train, axis=0)
y_train_no_5s = np.delete(y_train, indices_to_remove_train)
x_test_no_5s = np.delete(x_test, indices_to_remove_test, axis=0)
y_test_no_5s = np.delete(y_test, indices_to_remove_test)


# %%
show_dataset_min_and_max(x_train_no_5s, y_train_no_5s, features)

# %% [markdown]
# #### Normalized Values
# 

# %%
def normalize(x):
    mean = np.mean(x, axis = 0)
    std = np.std(x, axis = 0)

    normX = (x - mean) / std

    col_max = np.max(normX, axis=0)
    col_min = np.min(normX, axis=0)
    normX = np.divide(normX - col_min, col_max - col_min)

    return normX


x_train_n = normalize(x_train_no_5s)
y_train_n = normalize(y_train_no_5s)
x_test_n = normalize(x_test_no_5s)
y_test_n = normalize(y_test_no_5s)


# %%
show_dataset_min_and_max(x_train_n, y_train_n, features)

# %%
show_dataset_min_and_max(x_test_n, y_test_n, features)

# %%
print(y_train_n.shape)
print(y_test_n.shape)

# %% [markdown]
# ## Neural Network Design Experiments
# 

# %%
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from keras.layers import BatchNormalization
tf.random.set_seed(1234)

# %%
def initialize_sequential_model(layer_sizes, activation):
    """
    Function  to initialize a sequence of layers of a given size with a given activation function.
    The input layer has been fixed to a size of 8. 
    """
    model = Sequential()
    for size in layer_sizes:
        if not model.layers : 
            model.add(Dense(size, activation=activation, input_shape=(8,))) # for first layer
        elif size == 1:
            model.add(Dense(size, activation='linear')) # for last layer
        else:
            model.add(Dense(size, activation=activation)) # for every subsequent layer
    
    return model

def visualize_and_plot(labels_predicted, ground_truth, error_functions, history, save_filename):
    """ 
    Function to visualize the predicted regression values vs the actual ones
    and plot the error on each epoch in a single plot.
    """
    n = len(error_functions) if type(error_functions) is list else 1
    fig, ax = plt.subplots(1, n + 1, figsize=(12, 3))

    # Plot predicted vs actual values
    ax[0].plot(labels_predicted, ground_truth, '.', alpha=0.1)
    ax[0].plot(ground_truth, ground_truth)
    ax[0].set_title('Predicted vs Actual')
    ax[0].set_xlabel('Predicted values')
    ax[0].set_ylabel('Actual values')
    ax[0].legend(['pred', 'gr_tr'], loc='lower right')


    if type(error_functions) is list:
        for i in range(1, len(error_functions) + 1):
            error = error_functions[i - 1];
            # Plot training history
            ax[i].plot(history[error])
            ax[i].plot(history['val_{}'.format(error)])
            ax[i].set_title(error.upper())
            ax[i].set_ylabel(error)
            ax[i].set_xlabel('Epochs')
            ax[i].legend(['train', 'val'], loc='upper right')
    else:
        ax[1].plot(history[error_functions])
        ax[1].set_title(error_functions.upper())
        ax[1].set_ylabel(error_functions)
        ax[1].set_xlabel('Epochs')
        ax[1].legend(['train'], loc='upper right')

    plt.tight_layout()
    plt.savefig(f'figures\\{save_filename}.png')
    # plt.show()

def evaluate_model_on_final_epoch(history):
    evaluation_results = {}
    for error_function in history.keys():
        evaluation_results[error_function.upper()] = history[error_function][-1]
    print(evaluation_results)
    return evaluation_results

        

# %%
from tensorflow.keras.models import load_model
import pickle
import os

def create_and_train_model(*args):
    layer_sizes, activation_function, epochs, optimizer, loss_function, error_functions, model_name, dataset, load_from_file = args
    x_train_n, y_train_n = dataset

    model_file = '{}/{}.h5'.format('saved_models', model_name)
    history_file = '{}/{}_history.pkl'.format('saved_models', model_name)

    if load_from_file and os.path.exists(model_file) and os.path.exists(history_file):
        model = load_model(model_file)
        with open(history_file, 'rb') as file:
            history = pickle.load(file)
        print(f"Model and history loaded from '{model_name}'")
        print(history)
    else:
        model = initialize_sequential_model(layer_sizes, activation_function)

        model.compile(loss=loss_function, 
                    optimizer=optimizer,
                    metrics=error_functions
                    )
        history = model.fit(x_train_n[:,:],  # training data
                            y_train_n[:],     # Labels
                            epochs=epochs,
                            batch_size=128,
                            verbose=0,
                            validation_split=0.2
                            ).history
        print(f"New model created and trained: {model_file}")

        model.save(model_file)
        with open(history_file, 'wb') as file:
            pickle.dump(history, file)
        print(f"Model and history saved as '{model_file}'")

    return model, history



# %% [markdown]
# ### Evaluating Loss and Activation Functions

# %%
import json

def evaluate_architecture(layer_sizes, activation_functions, num_epochs, error_functions, dataset, file_name, load_from_file):

    x_train_n, y_train_n  = dataset 
    print("LAYER SIZES", layer_sizes)
    table = {}
    
    for activation_function in activation_functions:
        loss_function_table = {}
        print("====================================================================================")
        print('ACTIVATION FUNCTION: {}'. format(activation_function.upper()))
        for loss_function in error_functions:
            optimizer = tf.keras.optimizers.legacy.Adam()
            model_name = '{}-{}-{}'.format(file_name, activation_function, loss_function)
            m, h= create_and_train_model(layer_sizes, activation_function, num_epochs, optimizer, 
                                         loss_function, error_functions, model_name, 
                                         dataset, load_from_file)
            labels_predicted = m.predict(x_train_n)
            visualize_and_plot(labels_predicted, y_train_n, error_functions, h, model_name)
            metric_table = evaluate_model_on_final_epoch(h)
            loss_function_table[loss_function] = metric_table
        table[activation_function] = loss_function_table

    with open('json_outputs\\{}.json'.format(file_name), 'w') as file:
        json.dump(table, file, indent=4)

        

# %%
l = [16, 32, 64, 32, 16, 8, 4, 2, 1]
error_functions = ['mean_absolute_error', 'mean_squared_error', 'mean_absolute_percentage_error']
activation_functions = ['linear', 'relu', 'elu']

# %%
evaluate_architecture(l, activation_functions, 25, error_functions, [x_train_n, y_train_n], 'experiment_activation_loss', True)

# %%
evaluate_architecture(l, activation_functions, 25, error_functions, [x_train_n, y_train_n], 'experiment_activation_loss', True)

# %%
# from latex_export_functions import save_latex_table_from_json_activation_loss

# save_latex_table_from_json_activation_loss('experiment_activation_loss')

# %% [markdown]
# ### Evaluating Neural Network Layer Structures

# %%
activation_function = ['relu']
error_functions = ['mean_squared_error']
load_from_file = False

# %%
l1 = [16, 32, 64, 128, 256, 512, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
l2 = [16, 32, 16, 8, 1]
l3 = [16, 32, 64, 32, 16, 8, 4, 2, 1]
l4 = [16, 32, 64, 128, 256, 128, 64, 32, 16, 8, 4, 1]

# %%
evaluate_architecture(l1, activation_function, 200, error_functions, [x_train_n, y_train_n], 'experiment_size_{}'.format(l1), load_from_file)

# %%
evaluate_architecture(l2, activation_function, 200, error_functions, [x_train_n, y_train_n], 'experiment_size_{}'.format(l2), load_from_file)

# %%
evaluate_architecture(l3, activation_function, 200, error_functions, [x_train_n, y_train_n], 'experiment_size_{}'.format(l3), load_from_file)

# %%
evaluate_architecture(l4, activation_functions, 200, error_functions, [x_train_n, y_train_n], 'experiment_size_{}'.format(l4), load_from_file)

# # %%
# from latex_export_functions import create_latex_table_from_json_layer_sizes

# create_latex_table_from_json_layer_sizes('experiment_size_{}'.format(l1))
# create_latex_table_from_json_layer_sizes('experiment_size_{}'.format(l2))
# create_latex_table_from_json_layer_sizes('experiment_size_{}'.format(l3))
# create_latex_table_from_json_layer_sizes('experiment_size_{}'.format(l4))

# %% [markdown]
# ### Optimizer Evaluation

# %%
initial_learning_rates = [0.1, 0.01, 0.001]
decay_rates = [1, 0.96, 0.1]
num_epochs = 100
load_from_file = False

# %%
num_samples = len(x_train_n) * 0.8
batch_size = 128
number_of_steps_per_epoch = num_samples / batch_size
print('Number of steps per epoch', number_of_steps_per_epoch)

# %%
import json
from tensorflow.keras.optimizers import Adam, SGD

def evaluate_optimizer_and_learning_rate(initial_learning_rates, decay_rates, optimizer_name, num_epochs, dataset, file_name, load_from_file):

    x_train_n, y_train_n  = dataset 
    layer_sizes = [16, 32, 64, 128, 256, 128, 64, 32, 16, 8, 4, 1]
    activation_function = 'relu'
    loss_function = 'mean_squared_error'
    table = {}

    for initial_learning_rate in initial_learning_rates:
        learning_rate_table = {}
        for decay_rate in decay_rates:
            print('INITIAL LEARNING RATE: ', initial_learning_rate)
            print('DECAY RATE: ', decay_rate)
            model_name = "{}_{}_{}".format(optimizer_name, initial_learning_rate, decay_rate)
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=10000,
                decay_rate=decay_rate)
            optimizer = None 
            if optimizer_name == 'Adam':
                optimizer = Adam(learning_rate=lr_schedule)
            elif optimizer_name == 'SGD':
                optimizer = SGD(learning_rate=lr_schedule)
            elif optimizer_name == 'SGDM_Small':
                optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
            elif optimizer_name == 'SGDM_Large':
                optimizer = SGD(learning_rate=lr_schedule, momentum=0.99)
            else:
                raise TypeError("WE SHOULDN'T END HERE")

            m, h= create_and_train_model(layer_sizes, activation_function, num_epochs, optimizer, loss_function, error_functions, model_name, dataset, load_from_file)
            labels_predicted = m.predict(x_train_n)
            visualize_and_plot(labels_predicted, y_train_n, error_functions, h, model_name)
            metric_table = evaluate_model_on_final_epoch(h)

            learning_rate_table[decay_rate] = metric_table
        table[initial_learning_rate] = learning_rate_table
        


    with open('json_outputs\\{}.json'.format(file_name), 'w') as file:
        json.dump(table, file, indent=4)

        

# %%
evaluate_optimizer_and_learning_rate(initial_learning_rates, decay_rates, 'Adam', 200, [x_train_n, y_train_n], 'adam_learning_rate_table', load_from_file)

# %%
evaluate_optimizer_and_learning_rate(initial_learning_rates, decay_rates, 'SGD', 100, [x_train_n, y_train_n], 'sgd_learning_rate_table', load_from_file)

# %%
evaluate_optimizer_and_learning_rate(initial_learning_rates, decay_rates, 'SGDM_Small', 100, [x_train_n, y_train_n], 'sgd_small_learning_rate_table', load_from_file)

# %%
evaluate_optimizer_and_learning_rate(initial_learning_rates, decay_rates, 'SGDM_Large', 100, [x_train_n, y_train_n], 'sgd_large_learning_rate_table', load_from_file)

# %%
# from latex_export_functions import save_latex_table_json_optimizer

# save_latex_table_json_optimizer('adam_learning_rate_table')
# save_latex_table_json_optimizer('sgd_learning_rate_table')
# save_latex_table_json_optimizer('sgd_large_learning_rate_table')
# save_latex_table_json_optimizer('sgd_small_learning_rate_table')

# %% [markdown]
# ### Final Training and Plots

# %%
def create_and_train_final_model(*args):
    layer_sizes, activation_function, epochs, optimizer, loss_function, error_functions, model_name, dataset, load_from_file = args
    x_train_n, y_train_n = dataset

    model_file = '{}/{}.h5'.format('saved_models', model_name)
    history_file = '{}/{}_history.pkl'.format('saved_models', model_name)
    
    model = initialize_sequential_model(layer_sizes, activation_function)

    if load_from_file and os.path.exists(model_file) and os.path.exists(history_file):
        model = load_model(model_file)
        with open(history_file, 'rb') as file:
            history = pickle.load(file)
        print(f"Model and history loaded from '{model_name}'")
        print(history)
    else:
        model.compile(loss=loss_function, 
                    optimizer= optimizer,
                    metrics= error_functions
                    )
        history = model.fit(x_train_n[:,:], #training data
                        y_train_n[:],  #Labels
                        epochs=epochs,
                        batch_size=128,
                        verbose=0,
                        validation_split = 0
                    ).history
        print(f"New model created and trained: {model_file}")

        model.save(model_file)
        with open(history_file, 'wb') as file:
            pickle.dump(history, file)
        print(f"Model and history saved as '{model_file}'")  

    return model, history




def evaluate_full_dataset_architecture(layers, activation_function, optimizer, num_epochs, loss_function, dataset, test_set, file_name, load_from_file):
    x_train_n, y_train_n  = dataset 
    x_test_n, y_test_n = test_set
    model_name = 'final_model'
    error_functions = ['mean_squared_error', 'mean_absolute_error']
    m_train, h_train= create_and_train_final_model(layers, activation_function, num_epochs, optimizer, loss_function, error_functions, model_name, dataset, load_from_file)
    labels_predicted_train = m_train.predict(x_train_n)
    m_test, h_test= create_and_train_final_model(layers, activation_function, num_epochs, optimizer, loss_function, error_functions, model_name, test_set, True)
    labels_predicted_test = m_train.predict(x_test_n)

    visualize_and_plot(labels_predicted_train, y_train_n, loss_function, h_train, f'{model_name}_training_set')
    table_training = evaluate_model_on_final_epoch(h_train)

    visualize_and_plot(labels_predicted_test, y_test_n, loss_function, h_test, f'{model_name}_test_set')
    table_test = evaluate_model_on_final_epoch(h_test)


    with open('json_outputs\\{}-{}.json'.format(file_name, 'training_set'), 'w') as file:
        json.dump(table_training, file, indent=4)
    with open('json_outputs\\{}-{}.json'.format(file_name, 'test_set'), 'w') as file:
        json.dump(table_test, file, indent=4)

    # Plot predicted vs actual values

    fig, ax = plt.subplots(1, 2, figsize=(15,4))
    ax[0].plot(labels_predicted_train, y_train_n, '.', alpha=0.1)
    ax[0].plot(y_train_n, y_train_n)
    ax[0].set_title('Training Set')
    ax[0].set_xlabel('Predicted values')
    ax[0].set_ylabel('Actual values')
    ax[0].legend(['pred', 'gr_tr'], loc='lower right')

    ax[1].plot(labels_predicted_test, y_test_n, '.', alpha=0.1)
    ax[1].plot(y_test_n, y_test_n)
    ax[1].set_title('Test Set')
    ax[1].set_xlabel('Predicted values')
    ax[1].set_ylabel('Actual values')
    ax[1].legend(['pred', 'gr_tr'], loc='lower right')


    plt.savefig(f'figures\\{file_name}.png')


        

# %%
num_epochs = 200
layers = [16, 32, 64, 128, 256, 128, 64, 32, 16, 8, 4, 1]
activation_function = 'relu'
loss_function = 'mean_squared_error'  
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=10000,
                decay_rate= 0.96) 
optimizer = Adam(learning_rate=lr_schedule)
load_from_file = False
evaluate_full_dataset_architecture(layers, activation_function, optimizer, 200, loss_function, 
                                   [x_train_n, y_train_n], [x_test_n, y_test_n], 'final_training', load_from_file)


# %%
# from latex_export_functions import create_latex_table_from_json_final_training

# create_latex_table_from_json_final_training('final_training-training_set', 'final_training-test_set')

# %% [markdown]
# ### Change to Classification Neural Network

# %%
indices_to_remove_train = np.where(y_train >= 5)[0]
indices_to_remove_test = np.where(y_test >= 5)[0]

x_train_no_5s = np.delete(x_train, indices_to_remove_train, axis=0)
y_train_no_5s = np.delete(y_train, indices_to_remove_train)
x_test_no_5s = np.delete(x_test, indices_to_remove_test, axis=0)
y_test_no_5s = np.delete(y_test, indices_to_remove_test)

x_train_classification = normalize(x_train_no_5s)
x_test_classification = normalize(x_test_no_5s)

y_train_no_5s[y_train_no_5s < 2] = 0
y_train_no_5s[y_train_no_5s >= 2] = 1

y_test_no_5s[y_test_no_5s < 2] = 0
y_test_no_5s[y_test_no_5s >= 2] = 1

y_train_classification = y_train_no_5s
y_test_classification = y_test_no_5s

dataset_classification_train = [x_train_classification, y_train_classification]

# %%
def visualize_and_plot_classification(history, test_set, save_filename):
    """ 
    Function to visualize the predicted regression values vs the actual ones
    and plot the error on each epoch in a single plot.
    """

    error = 'binary_accuracy';
    plt.plot(history[error])
    if not test_set:
        plt.plot(history['val_{}'.format(error)])

    plt.title(error.upper())
    plt.ylabel(error)
    plt.xlabel('Epochs')
    plt.legend(['train', 'val'], loc='upper right')

    plt.tight_layout()
    plt.savefig(f'figures\\{save_filename}.png')
    plt.show()

# %%
def create_and_train_classification_model(*args):
    layer_sizes, activation_function, epochs, optimizer, loss_function, output_function, model_name, dataset, validation_split, load_from_file = args
    x_train_n, y_train_n = dataset

    model_file = '{}/{}.h5'.format('saved_models', model_name)
    history_file = '{}/{}_history.pkl'.format('saved_models', model_name)
    
    model = initialize_sequential_model(layer_sizes, activation_function)
    model.add(Dense(1, activation=output_function))

    if load_from_file and os.path.exists(model_file) and os.path.exists(history_file):
        model = load_model(model_file)
        with open(history_file, 'rb') as file:
            history = pickle.load(file)
        print(f"Model and history loaded from '{model_name}'")
        print(history)
    else:
        model.compile(loss=loss_function, 
                    optimizer= optimizer,
                    metrics= ['binary_accuracy']
                    )
        history = model.fit(x_train_n[:,:], #training data
                        y_train_n[:],  #Labels
                        epochs=epochs,
                        batch_size=128,
                        verbose=0,
                        validation_split = validation_split
                    ).history
        print(f"New model created and trained: {model_file}")

        model.save(model_file)
        with open(history_file, 'wb') as file:
            pickle.dump(history, file)
        print(f"Model and history saved as '{model_file}'")  

    return model, history




def evaluate_classification_dataset_architecture(layers, activation_function, optimizer, num_epochs, loss_function, output_function, dataset, validation_split, file_name, load_from_file):
    model_name = file_name
    m, h= create_and_train_classification_model(
        layers, activation_function, num_epochs, optimizer, loss_function, output_function, 
        model_name, dataset, validation_split, load_from_file)
    
    visualize_and_plot_classification(h, validation_split == 0, model_name)
    table = evaluate_model_on_final_epoch(h)

    with open('json_outputs\\{}.json'.format(file_name), 'w') as file:
        json.dump(table, file, indent=4)

    return m, h

# %%
num_epochs = 200
layers = [16, 32, 64, 128, 256, 128, 64, 32, 16, 8, 4]
activation_function = 'relu'
loss_function = 'binary_crossentropy'
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=10000,
                decay_rate= 0.96) 
optimizer=Adam(learning_rate=lr_schedule)
load_from_file = False


evaluate_classification_dataset_architecture(layers, activation_function, optimizer, num_epochs, loss_function, 'sigmoid', 
                                             dataset_classification_train, 0.2, 'classificaiton_retraining_200', load_from_file )


# # %%
# from latex_export_functions import create_latex_table_for_classification_validation

# create_latex_table_for_classification_validation('classificaiton_retraining_200')
# create_latex_table_for_classification_validation('classificaiton_retraining_100')

# %%
optimizer=Adam(learning_rate=lr_schedule)
evaluate_classification_dataset_architecture(layers, activation_function, optimizer, 100, loss_function, 'sigmoid', 
                                             dataset_classification_train, 0.2, 'classificaiton_retraining_100', load_from_file )


# %%
optimizer=Adam(learning_rate=lr_schedule)
classification_model, classification_history = evaluate_classification_dataset_architecture(layers, 'relu', optimizer, 100, loss_function, 'sigmoid', 
                                             dataset_classification_train, 0, 'classification_training_no_validation_set', load_from_file )

# %%
loss_train, accuracy_train = classification_model.evaluate(x_train_classification, y_train_classification)
print(f'Binary Crossentropy Loss on Training Set: {loss_train}')
print(f'Accuracy on Training Set: {accuracy_train}')

# %%
loss_test, accuracy_test = classification_model.evaluate(x_test_classification, y_test_classification)
print(f'Binary Crossentropy Loss on Test Set: {loss_test}')
print(f'Accuracy on Test Set: {accuracy_test}')


