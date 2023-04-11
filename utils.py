from imports import *

def convert_to_sg_ch(x):
    if x.ndim==1:
        return x
    else:
        return np.mean(x, axis=1, dtype='float32')

def normalization(x):
    
    x_norm = x/max(np.abs(x))
    
    return x_norm #norm [-1,1]

def standardization(x):
    return (x-np.mean(x))/np.std(x)

def amplitude(W_signal):
    return np.abs(W_signal)

def phase(W_signal):
    return np.angle(W_signal)

def sigwin(x, l, w_type, overlap):
    """
    w_type[string] can be:  -rect
                            -boxcar
                            -triang
                            -blackman
                            -hamming
                            -hann
                            -bartlett
                            -flattop
                            -parzen
                            -bohman
                            -blackmanharris
                            -nuttall
                            -barthann

    overlap [percentage]
    l[sample number]
    x[list or np.array]
    """
    overlap=overlap/100
    if type(x)==np.ndarray:
        x=x.tolist()
    w = []
    delay = int((1- overlap)*l)

    if( w_type !='rect'):
        win = windows.get_window(w_type,l).tolist()

    for i in range(0, len(x), delay):
        if i+l<=len(x):
            if (w_type == 'rect'):
                w.append(x[i:i+l])
            else:
                w.append(np.multiply(win,x[i:i+l]))

    return np.array(w)


def sigrec(w_signal, overlap, mode='MEAN'):
    """
    Arguments:

        w_signal: an array with the windows of size #windows x window_length

        overlap: the percentage of overlapping between windows
        
        mode: method to reconstruct the signal:
		'OLA' for overlap and addition
		'MEAN' for overlap and mean (default if not 'OLA')

    Outputs:


        x: the reconstructed signal of size signal_length


    """

    n = len(w_signal)  # number of windows
    overlap = overlap / 100  # calc percentage
    l = len(w_signal[0])  # window len

    non_ov = int((1 - overlap) * l)  # non overlapping section of 2 windows
    lenx = (n - 1) * non_ov + l  # len of signal to reconstruct. formula might be wrong.
    delay = non_ov  # used to delay i'th window when creating the matrix that will be averaged

    w_frm_aux = np.zeros((n, lenx), dtype ='float32')  # size = windows x signal_length
							# dtype='float32' to reduce memory usage
    for i in range(0, len(w_signal)):
        crt = np.zeros(i * delay).tolist()
        crt.extend(w_signal[i])
        crt.extend(np.zeros(lenx - i * (delay) - l).tolist())

        w_frm_aux[i] += crt

    summ = np.sum(w_frm_aux, axis=0)
    if mode == 'OLA': return summ
    
    nonzero = w_frm_aux != 0
    divvect = np.sum(nonzero, axis=0)   
    divvect[divvect==0]=1   #avoid division by zero
    x = summ / divvect

    return x
    
def DWT(w_signal, wavelet_type):
    '''
    wavelet_type can be:
        dbX, where X is a number in [1;38]
        symX, where X is a number in [2;20]
        coifX, where X is a number in [1;17]
        biorX, where X can be: (1.1, 1.3, 1.5,
                                2.2, 2.4, 2.6, 2.8,
                                3.1, 3.3, 3.5, 3.7, 3.9,
                                4.4, 5.5, 6.8)
        rbioX where X can be: (1.1, 1.3, 1.5,
                                2.2, 2.4, 2.6, 2.8,
                                3.1, 3.3, 3.5, 3.7, 3.9,
                                4.4, 5.5, 6.8)
        haar,
        dmey.
    '''
    W_signal, _ = pywt.dwt(w_signal, wavelet_type, axis=1)
    return np.asanyarray(W_signal)

def FFT(w_signal, N_fft):
    
    """
    Arguments:


        w_signal: an array of windows from a signal, of size #windows x window_length


        N_fft:  #of points to perform FFT

    Outputs:


        W_signal: an array containing the FFT of each window, of size #windows x N_fft

    """


    W_signal=np.fft.fft(w_signal,N_fft, axis = 1)

  
    return np.array(W_signal)

def IFFT(W_signal, l):
    
    """
    Arguments:

    W_signal: an array containing the FFT of each window, of size #windows x N_fft
    l: length of each window in the output array of windows
    
    
    Outputs:

    w_signal: an array of windows from a signal, of size #windows x window_length
    """
    
    w_signal=np.fft.ifft(W_signal, W_signal.shape[1], axis = 1)[:, 0:l]
    
    return w_signal

def autocorrelation(w_signal, N_fft):
    window_length = np.shape(w_signal)[1]
    autocorrel = IFFT(np.square(amplitude(FFT(w_signal, N_fft))), N_fft)[:, 0:window_length]
    return autocorrel.real

def cepstrum(w_signal, N_fft):
    '''
    Uses FFT, IFFT and log functions to calculate the Cepstrum
    '''
    window_length = np.shape(w_signal)[1]
    C = IFFT(np.log(amplitude(FFT(w_signal, N_fft))), N_fft)[:, 0:window_length]
    return C.real
    
def next_pow_of_2(x):
    return int(np.power(2, np.ceil(np.log(x)/np.log(2))))

def MSE(song, reconstructed):
    '''
    MSE returns a tuple of 2 elements:
    	1. MSE value of the entire signal
    	2. MSE value in each point
    '''
    diff = song[0:len(reconstructed)] - reconstructed
    return (np.mean(diff**2), (diff**2))

def extract_features(w_signal, feature_extractor, feature_dict, feature_list, window=True, variance_type='var'):
    '''
    Arguments:
        - w_signal [2D-array], windowed signal
        - feature_extractor [FeatureExtractor object], used to extract the feature
        - feature_dict [dict], parameters needed for the extraction of features, e.g. features_dict['n_fft'] = 1024
        - feature_list [list], list of features to be extracted
        - window [boolean], if true extract features window by window
        - variance_type [string], either var for variance or smad for square median absolute deviation
    Output:
        - W_feature [2D-array]
    '''
    W_feats = []
    if window:
        for win in w_signal:
            _ = feature_extractor.extract_features(win, ['spect', 'mel_spect'], feature_dict)
            features = feature_extractor.extract_features(win, feature_list, feature_dict, variance_type)
            W_feats.append(features)

    else:
        features = feature_extractor.extract_features(w_signal, feature_list, feature_dict, variance_type)
        W_feats.extend(features)

    W_feats = np.asanyarray(W_feats)
    return W_feats

def save_features(path, music_genre, song_name, W_features, tfrecord, writer):
    '''
    If tfrecord is set to True:
    	The script adds the requested features in the Feature.tfrecord file
    Else:
    	The script creates (or overwrites) a song_name.npy file which contains the requested features
    	of the selected .wav file.

    Arguments:
        - path [string], relative path of the Features folder
        - music_genre [string], name of the music genre
                                must be one of the genres in the database!
        - song_name [string], name of the song
        - W_features [2D-array], array of features to be writen, size #windows x #features
        - tfrecord [bool], if true saves W_features in the '.tfrecord' file
                           else it saves it as song_name.npy file.
        - writer [tf.io.TFRecordWriter], the writer used to write records to a .tfrecord file.
    '''
    if music_genre.lower() != song_name.split('.')[0]:
        raise Exception("Song name does't match its parent folder's name")
        
    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 
              'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

    if not os.path.exists(path):
        os.mkdir(path) #create Features folder if does not already exist

    if music_genre not in genres:
        raise Exception(music_genre + ' is not a genre in the GTZAN database')

    if tfrecord:
        windows_label = genres.index(music_genre) * np.ones(W_features.shape[0], dtype=np.int64)
        labels = np.zeros([W_features.shape[0], len(genres)], dtype=np.int64)
        labels[np.arange(W_features.shape[0]), windows_label] = 1
        convert_to_tfrecord(writer, W_features, labels)
    else:
        music_path = os.path.join(path, music_genre)
        if not os.path.exists(music_path):
            os.mkdir(music_path) #create song genre folder if does not already exist
        if song_name[-4:len(song_name)] == '.wav':
            song_name = song_name[0:-4]
        file = os.path.join(music_path, song_name)
        np.save(file, W_features)

def max_freq(W_signal, sample_rate):
    """
    This function returns the maximum frequency of a windowed Fourier transformed signal.
    
    W_signal: an array containing the FFT of each window, of size #windows x N_fft // 2


    sampling_rate: the frequency used to sample the song
    
    """
   
    f_max = (sample_rate/(2*W_signal.shape[1]))*np.max(np.argmax(W_signal,axis=1))
    
    return f_max


def genre_maximum_frequency(path, window_length, window_type, overlap):
    """
    Function that finds the maximum frequency for each musical genre in the GTZAN dataset, as well as the overall
    maximum frequency.

    Arguments:
        -path: the relative path to the GTZAN dataset
        -window_length: the length of the window (in samples) e.g. 11025 samples = 0.5s
        -window_type: the type of window used to split the signal, e.g. 'rect'
        -overlap: the percentage of overlapping between windows, e.g. 50%

    Outputs:
        -genre_freq: a dictionary containing all the maximum frequencies for each musical genre.
    """
    genre_freq = {}

    for genre in os.listdir(path):
        genre_max = 0
        for file in os.listdir(os.path.join(path, genre)):
            sample_rate, song = wavfile.read(os.path.join(path, genre, file))
            song = convert_to_sg_ch(song)
            song = normalization(song)
            w_signal = sigwin(song, window_length, window_type, overlap)
            W_FFT = FFT(w_signal, next_pow_of_2(window_length))
            W_FFT = amplitude(W_FFT)[:, 0:W_FFT.shape[1]//2]

            f_max = max_freq(W_FFT, sample_rate)
            if genre_max < f_max:
                genre_max = f_max

        genre_freq[genre] = genre_max

    highest_freq = max(genre_freq.values())
    genre_freq["Highest frequency"] = highest_freq

    return genre_freq

def get_callbacks(path_model):
    logdir = '../Log/log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)
    model_checkpoint = ModelCheckpoint(path_model, monitor='val_loss', verbose=1, save_best_only=True)
    return [model_checkpoint, tensorboard_callback]

def create_model(no_features, no_classes, optimizer, dropout_rate=0.5, summary=True):
    input = Input(shape=(no_features,))
#    bn0 = BatchNormalization()(input)
    hdn1 = Dense(512, name='layer1')(input)
    act1 = Activation('relu')(hdn1)
    # bn1 = BatchNormalization()(act1)
    dp1 = Dropout(dropout_rate)(act1)

    hdn2 = Dense(256, name='layer2')(dp1)
    act2 = Activation('relu')(hdn2)
    # bn2 = BatchNormalization()(act2)
    dp2 = Dropout(dropout_rate)(act2)

    hdn3 = Dense(128, name='layer3')(dp2)
    act3 = Activation('relu')(hdn3)
    # bn3 = BatchNormalization()(act3)
    dp3 = Dropout(dropout_rate)(act3)

    hdn4 = Dense(64, name='layer4')(dp3)
    act4 = Activation('relu')(hdn4)
    # bn4 = BatchNormalization()(act4)
    dp4 = Dropout(dropout_rate)(act4)

    hdn5 = Dense(32, name='layer5')(dp4)
    act5 = Activation('relu')(hdn5)
    # bn5 = BatchNormalization()(act5)
    dp5 = Dropout(dropout_rate)(act5)

    hdn6 = Dense(no_classes)(dp5)
    output = Activation('softmax')(hdn6)
    model = Model(inputs=input, outputs=output)

    if summary:
        print(model.summary())

    model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    return model

def train_model(model, tfrecord, train_dataset, val_dataset, batch_size, epochs, path_model):
    '''
    If tfrecord is True:
        train_dataset and val_dataset must be a tensorflow.data.Dataset.Batch
    Else:
        train_dataset and val_dataset must be a tuple of (features, labels)
    '''
    if tfrecord:
        model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=2, callbacks=get_callbacks(path_model=path_model))
    else:
        model.fit(x=train_dataset[0], y=train_dataset[1], validation_data=(val_dataset[0], val_dataset[1]), 
                  batch_size=batch_size, epochs=epochs, verbose=2, callbacks=get_callbacks(path_model=path_model))
            

def test_model(model, dataset):
    return model.predict(dataset)
    
def write_cardinality(path, cardinality):
    '''
    Write the cardinality of a TFRecordDataset in a ".txt" file.
    
    Arguments:
        - path [string], relative path to the ".txt" file
        - cardinality [int], the length of the TFRecordDataset
        
    Raises:
        - TypeError if cardinality is not an integer
    '''
    if type(cardinality)!=int:
        raise TypeError("Cardinality is not an integer")
    else:
        dir_path = os.path.join(*path.split(os.sep)[0:-1])
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        f = open(path, 'w')
        f.write(str(cardinality))
        f.close()

def load_cardinality(path):
    '''
    Arguments:
        - path [string], relative path to the ".txt" file
        
    Output:
        - cardinality [int], the length of the TFRecordDataset
    '''
    f = open(path, 'r')
    cardinality = f.read()
    f.close()
    return int(cardinality)

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse_window_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    #example_proto needs to be a string scalar tensor

    song_feature_description = {
    'window': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    }
    
    return tf.io.parse_single_example(example_proto, song_feature_description)

def parse_and_decode_function(example_proto):
    feature_description = {
        'window': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    element = tf.io.parse_single_example(example_proto, feature_description)
    decoded_label = tf.io.decode_raw(element["label"], "int64")
    decoded_window = tf.io.decode_raw(element["window"], "float64")
    return (decoded_window, decoded_label)

def convert_to_tfrecord(writer, W_features, labels ):

    window_dict = {}
    for i in range(W_features.shape[0]):
        window_dict["window"] = bytes_feature(W_features[i].tobytes())
        window_dict["label"] = bytes_feature(labels[i].tobytes())

        ftexample = tf.train.Example(features=tf.train.Features(feature=window_dict) )
        ftserialized = ftexample.SerializeToString()
        writer.write(ftserialized)

def prep_dataset(dataset, batch_size,shuffle_buffer, shuffle_seed = 5):
    '''
    A function that prepares the TFRecordDataset for the neural network model.
    This function should implement the mapping of each element in the dataset, followed by a shuffle and a group by batch.

    Arguments:
        - dataset: the dataset on which to perform the dataset.shuffle() and dataset.batch() methods
        - batch_size: the size of a batch used to train the model.
        - shuffle_buffer: the size of the buffer used to shuffle the data.
        - shuffle_seed: the value of the random seed that will be passed to the dataset.shuffle() function

    Outputs:
        - dataset: the TFRecordDataset to be used in the neural network.
    '''
    decoded_batch_dataset = dataset.shuffle(shuffle_buffer, shuffle_seed, False).batch(batch_size, False)
    return decoded_batch_dataset
        
def squared_median_abs_dev(x):
    if len(x.shape) == 1:
        return scipy.stats.median_absolute_deviation(x)**2
    elif len(x.shape) == 2:
        return np.mean(scipy.stats.median_absolute_deviation(x, axis=1)**2)
    else:
        raise TypeError("Input must be a vector or a matrix")
    
def plot_confusion_matrix(y_true, y_pred):
    cm = sklearn.metrics.confusion_matrix(y_pred=y_pred, y_true=y_true)
    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels=genres).plot()
    
    
def compute_scaller(tfrecord):
    '''
    Computes the scaller on the entire database.
    
    Arguments:
        - tfrecord [boolean], if true, reads data from TFRecord files, else from .npy files
    Output:
        - scaller [a fitted sklearn.preprocessing scaller]
    '''
    X = []
    scaller = StandardScaler()
    if tfrecord:
        paths = [os.path.join('..', 'TFRecord', 'Train_Features.tfrecord'), os.path.join('..', 'TFRecord', 'Test_Features.tfrecord')]
        raw_dataset = tf.data.TFRecordDataset(paths[0])
        train = raw_dataset.map(parse_and_decode_function)
        raw_dataset = tf.data.TFRecordDataset(paths[1])
        test = raw_dataset.map(parse_and_decode_function)
        combined_dataset = train.concatenate(test)
        for element in combined_dataset:
            X.append(element[0].numpy())
        scaller.fit(X)
    else:
        paths = [os.path.join('..', 'Train'), os.path.join('..', 'Test')]
        for path in paths:
            genre_folders = sorted(os.listdir(path))
            for genre in genre_folders:
                files = sorted(os.listdir(os.path.join(path, genre)))
                for file in files:
                    npy = np.load(os.path.join(path, genre, file))
                    X.extend(npy)
        scaller.fit(X)
    return scaller
        
def create_test_npy(path, window_length, scaller):
    '''Creates x_test and y_test (true labels) from .npy files'''
    no_of_windows = 30 * 22050 // window_length
    genres_label = {'Blues': 0, 'Classical': 1, 'Country': 2, 'Disco': 3, 'Hiphop': 4, 'Jazz': 5, 'Metal': 6,
                    'Pop': 7,'Reggae': 8, 'Rock': 9}
    genre_folders = sorted(os.listdir(path))
    x_test, y_true = [], []
    for genre in genre_folders:
        files = sorted(os.listdir(os.path.join(path, genre)))
        for file in files:
            npy = np.load(os.path.join(path, genre, file))
            x_test.extend(npy)
            for _ in range(no_of_windows):
                y_true.append(genres_label[genre])
    x_test = scaller.transform(x_test)
    return (np.asanyarray(x_test), np.asanyarray(y_true))
   
def create_test_tfrecord(path, scaller):
    '''Gets x_test and y_true (true labels) for test from '''
    raw_dataset = tf.data.TFRecordDataset(os.path.join(path))
    parsed_dataset = raw_dataset.map(parse_and_decode_function)
    x_test, y_test , original_labels = [], [], []
    for element in parsed_dataset:
        x_test.append(element[0].numpy())
        y_test.append(np.argmax(element[1].numpy()))
        original_labels.append(element[1])
    x_test = scaller.transform(x_test)
    return (np.asanyarray(x_test), np.asanyarray(y_test))

def get_per_class_accuracy(y_true, y_pred):
    cm = sklearn.metrics.confusion_matrix(y_pred=y_pred, y_true=y_true)
    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    print("\nPer class accuracy:")
    for line, (i,genre) in zip(cm, enumerate(genres)):
        print("\tAccuracy of class", genre, "is:", format(100*line[i]/np.sum(line), ".2f"), "%")
    print('\n')

def plot_roc_curve(y_true, y_score):
    fpr, tpr, roc_auc = dict(), dict(), dict()
    colors = ['purple','red', 'black', 'green', 'blue','chocolate', 'lime', 'orange', 'magenta','cyan']
    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    plt.figure()
    y_test = to_categorical(np.asanyarray(y_true), num_classes=10)
    for i in range(len(genres)):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], label='ROC curve for ' + genres[i] + ' (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
