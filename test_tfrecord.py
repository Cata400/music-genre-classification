from utils import *
from k_fold import k_fold
from split_dataset import split_dataset
from iterate_dataset import iterate_dataset


def _test_window(window_length, window_type, overlap, feature_type, normalize, standardize, no_of_songs_to_test_per_genre, *args):
    raw_dataset = tf.data.TFRecordDataset(os.path.join('..', 'TFRecord', 'Features.tfrecord'))
    parsed_dataset = raw_dataset.map(parse_and_decode_function)
    num_of_windows = load_cardinality(os.path.join("..", "Cardinality", "Features.txt")) // 1000
    cnt_not_identical = 0
    cnt_wrong_genre = 0
    last_skip = 0
    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    songs_for_test = []
    for genre in genres:
        songs_for_test.append(sorted(random.sample(range(len(os.listdir('..' + os.sep + 'GTZAN' + os.sep + genre))), no_of_songs_to_test_per_genre)))

    for genre in enumerate(songs_for_test):
        for song_no in genre[1]:
            song_genre = genres[genre[0]]
            tf_song_skipper = 100 * genre[0] * num_of_windows + song_no * num_of_windows
            song_path = os.path.join("..", "GTZAN", song_genre, song_genre.lower() + "." + str(song_no).zfill(5) + ".wav")
            (rate, x) = wavfile.read(song_path)
            x = convert_to_sg_ch(x)
            if normalize == True:
                x = normalization(x)
            if standardize == True:
                x = standardization(x)
            w_list = sigwin(x, window_length, window_type, overlap)
            W_feats = extract_features(w_list, feature_type, *args)

            if W_feats.shape[0] > num_of_windows:
                W_feats = W_feats[0:num_of_windows]
            else:
                while (W_feats.shape[0] < num_of_windows):
                    non_overlap_dim = int((1 - overlap / 100) * W_feats.shape[1])
                    new_wind = W_feats[-1, non_overlap_dim:W_feats.shape[1]]
                    zeros = np.zeros(non_overlap_dim)
                    new_wind = np.concatenate((new_wind, zeros))
                    new_wind = np.reshape(new_wind, (1, new_wind.shape[0]))
                    W_feats = np.append(W_feats, new_wind, axis=0)
            win_no = 0
            parsed_dataset = parsed_dataset.skip(tf_song_skipper - last_skip)
            for (i, window), tf_window in zip(enumerate(W_feats), parsed_dataset):
                if not np.allclose(tf_window[0], window):
                    cnt_not_identical += 1
                    win_no += 1
                    print("WINDOW IS NOT IDENTICAL! window:", win_no, "song:", song_no, "genre:", song_genre)
                    print("Expected:", window)
                    print("Got:", tf_window[0])
                else:
                    if song_genre != genres[list(tf_window[1]).index(1)]:
                        print("Genre does not match")
                        cnt_wrong_genre += 1
                    win_no += 1
                    #print("Same window:", win_no, "song:", song_no, "genre:", song_genre)
                if i == num_of_windows:
                    break
            last_skip = tf_song_skipper
    print("Finished checking")
    if cnt_not_identical or cnt_wrong_genre:
        print("Number of not identical windows found:", cnt_not_identical)
        print("Number of wrong genres:", cnt_wrong_genre)
    else:
        print("All tested songs have similar windows")


def _check_genre_distribution(path, dataset, *args):
    '''
    Arguments:
        - path [os.path], relative path to the .tfrecord file, either Test or Validation
    or
        - dataset [tf.data.Dataset], a dataset to check it's distribution
    !The unused argument must an empty list!
        - *args, split_perc, if path is given
                 k and fold_nr, if validation dataset is given
                 empty list, if train dataset in given
    '''
    if dataset == []:
        raw_dataset = tf.data.TFRecordDataset(path)
        parsed_dataset = raw_dataset.map(parse_and_decode_function)
    if path == []:
        parsed_dataset = dataset
    num_of_windows = load_cardinality(os.path.join("..", "Cardinality", "Features.txt")) // 1000
    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
    genres_counter = {'Blues': 0, 'Classical': 0, 'Country': 0, 'Disco': 0, 'Hiphop': 0, 'Jazz': 0, 'Metal': 0,
                    'Pop': 0, 'Reggae': 0, 'Rock': 0}

    for element in parsed_dataset:
        index = np.argmax(element[1].numpy())
        genres_counter[genres[index]] += 1

    for key in genres_counter:
        genres_counter[key] //= num_of_windows

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(genres_counter)), list(genres_counter.values()), align="center")
    plt.xticks(range(len(genres_counter)), list(genres_counter.keys()))
    if dataset == []:
        plt.title("Song distribution in " + os.path.split(path)[-1] + "\nSplit between: Train = " + str(args[0]) +
                  "% and Test = " + str(100 - args[0]) + "%")
    elif args[0] != []:
        plt.title("Song distribution in validation dataset\nk = " + str(args[0]) + ", fold_nr = " + str(args[1]))
    else:
        plt.title("Song distribution in train dataset")
    plt.show()
    print("Song distribution is:", genres_counter)

def test_tfrecord(path, window_length, window_type, overlap, feature_type,normalize, standardize, tfrecord, split_perc, k, no_of_songs_to_test_per_genre, *args):
    '''
    Arguments:
        - path [os.path], the path of the GTZAN dataset relative to the source file's folder
        - window_length [int], length of a signal window (in samples)
        - window_type [string], the type of window used to split the signal, e.g. 'blackman', 'hamming', 'hanning', etc.
        - overlap [int], the percentage of overlapping between windows
        - feature_type [string], the feature to be extracted from a signal window.
                                Can be: FFT, DWT, autocorrelation, cepstrum
        - normalize [boolean], decides if data will be normalized default False
        - standardize [boolean], decides if data will be standardized default False
        - tfrecord [boolean], indicates whether the .tfrecord conversion should apply.
        - split_perc [int], percentage used to split Features in (split_perc) Train
                            and (100 - split_perc) Test
        - k [int], number of fold for the K-Fold algorithm
        - *args: the specific parameters required for the feature, e.g N_fft, 'wavelet_type', etc
    '''
    print("Creating Features.tfrecord...")
    iterate_dataset(path, window_length, window_type, overlap, feature_type, normalize, standardize, tfrecord, *args)
    print("Features.tfrecord successfully created.")
    print("\nChecking the tfrecord file...")
    _test_window(window_length, window_type, overlap, feature_type, normalize, standardize, no_of_songs_to_test_per_genre, *args)
    print("\nSpliting Features.tfrecord in Train_Features.tfrecord and Test_Features.tfrecord...")
    split_dataset(path, split_perc, load_cardinality(os.path.join("..", "Cardinality", "Features.txt")), tfrecord)
    print("Train_Features.tfrecord and Test_Features.tfrecord successfully created.")
    print("\nChecking distribution of songs in Test_Features...")
    _check_genre_distribution(os.path.join("..", "TFRecord", "Test_Features.tfrecord"), [], split_perc)
    print("Graph generated")
    fold = random.sample(range(k), 1)[0] + 1
    print("\nApplying K-Fold algorithm with k =",k ,", fold_nr =", fold)
    train, val = k_fold(path, k, fold, tfrecord, load_cardinality(os.path.join("..", "Cardinality", "Train_Features.txt")))
    print("Succesfully applied K-Fold algorithm")
    print("\nPreparing validation dataset...")
    num_of_windows = load_cardinality(os.path.join("..", "Cardinality", "Features.txt")) // 1000
    val = prep_dataset(val, num_of_windows , 5 * num_of_windows).unbatch()
    print("Validation dataset is prepared.")
    print("\nChecking distribution of songs in Validation dataset...")
    _check_genre_distribution([], val, k, fold)
    print("Graph generated")
    print("\nPreparing train dataset...")
    train = prep_dataset(train, num_of_windows , 5 * num_of_windows).unbatch()
    print("Train dataset is prepared.")
    print("\nChecking distribution of songs in Train dataset...")
    _check_genre_distribution([], train, [])
    print("Graph generated")

    print("\nEND OF TESTING")
