import numpy as np
import pandas as pd
import re




#Load training and testing data directly from CSV file
def load_training_and_testing_data(train_file, test_file):
    train_data = pd.read_csv(train_file,header = None)
    train_images = train_data.iloc[:,1:].values
    train_images = train_images.astype(np.float)

    train_labels_flat = train_data.iloc[:,0:1].values
    train_labels_count = np.unique(train_labels_flat).shape[0]
    train_labels = dense_to_one_hot(train_labels_flat, train_labels_count)
    train_labels = train_labels.astype(np.uint8)


    test_data = pd.read_csv(test_file,header = None)
    test_images = test_data.iloc[:,1:].values
    test_images = test_images.astype(np.float)

    test_labels = test_data.iloc[:,0:1].values
    test_labels = dense_to_one_hot(test_labels, train_labels_count)
    test_labels = test_labels.astype(np.uint8)

    print(train_labels_count)

    return train_images, train_labels, test_images, test_labels, train_labels_count



# Convert class labels from scalars to one-hot vectors 
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
