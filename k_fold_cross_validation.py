from k_fold import k_fold
from utils import *




def k_fold_cross_validation(path, k, path_model, tfrecord, train_cardinality, batch_size, shuffle_buffer, 
                            epochs, optimizer, dropout, scaller, shuffle_mode):
    '''
    Trains the model using K-fold algorithm
    
    Arguments:
        - path [string], relative path to Train folder
        - k [int], number of subsets used to split the Train set into
        - path_model [string], relative path to save the trained model
        - tfrecord [boolean], indicates whether to train the model using the .tfrecord files
        - train cardinality [int], the length of the Train.tfrecord file
        - batch_size [int], the size of a batch used to train the model
        - shuffle_buffer [int], the size of the buffer used to shuffle the data
    '''
    if not os.path.exists(os.path.split(path_model)[0]): #create Model folder if it does not exist
        os.mkdir(os.path.split(path_model)[0])
    if tfrecord:
        for i in range(k):
            train_dataset, val_dataset = k_fold(path, k, i+1, tfrecord, train_cardinality)
            x_train, y_train, x_val, y_val = [], [], [], []
            for element in train_dataset:
                x_train.append(element[0].numpy())
                y_train.append(element[1].numpy())
            for element in val_dataset:
                x_val.append(element[0].numpy())
                y_val.append(element[1].numpy())
            x_train = scaller.transform(x_train)
            x_val = scaller.transform(x_val)
            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) 
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)) 
            for element in train_dataset.take(1):
                no_of_features = int(tf.shape(element[0])[0])
                no_of_classes = int(tf.shape(element[1])[0])
            if shuffle_mode:
                train_length = 0
                for _ in train_dataset:
                    train_length += 1
                combined_dataset = train_dataset.concatenate(val_dataset)
                combined_dataset = combined_dataset.shuffle(train_cardinality)
                train_dataset = combined_dataset.take(train_length)
                combined_dataset = combined_dataset.skip(train_length)
                val_dataset = combined_dataset.take(-1) #take remaining tensors
                train_dataset = prep_dataset(train_dataset, batch_size, shuffle_buffer)
                val_dataset = prep_dataset(val_dataset, batch_size, shuffle_buffer)
            else:
                train_dataset = prep_dataset(train_dataset, batch_size, shuffle_buffer)
                val_dataset = prep_dataset(val_dataset, batch_size, shuffle_buffer)
            model = create_model(no_of_features, no_of_classes, optimizer, dropout, summary=True)
            train_model(model, tfrecord, train_dataset, val_dataset, batch_size, epochs, path_model)
            break #only trains for one fold, will be updated
    else:
        for i in range(k):
            x_train, x_val, y_train, y_val = k_fold(path, k, i+1, tfrecord, train_cardinality)
            x_train = scaller.transform(x_train)
            x_val = scaller.transform(x_val)
            if shuffle_mode:
                train_length = x_train.shape[0]
                data = np.concatenate((x_train, x_val), axis=0)
                y_data = np.concatenate((y_train, y_val), axis=0)
                data_shuff, y_data_shuff = shuffle(data, y_data, random_state=42)
                x_train = data_shuff[0:train_length, :]
                x_val = data_shuff[train_length:len(data_shuff),:]
                y_train = y_data_shuff[0:train_length, :]
                y_val = y_data_shuff[train_length:len(data_shuff),:]
            else:
                x_train, y_train = shuffle(x_train, y_train, random_state=42)
                x_val, y_val = shuffle(x_val, y_val, random_state=42)
            model = create_model(x_train.shape[1], y_train.shape[1], optimizer, dropout, summary=True)
            train_model(model, tfrecord, (x_train, y_train), (x_val, y_val), batch_size, epochs, path_model)
            break #only trains for one fold, will be updated