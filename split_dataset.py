from utils import *


def split_dataset(path, split_perc, features_cardinality, tfrecord=True):
    '''
    Script that splits the dataset in Train and Test features.

    Arguments:
        - path [string], relative path to the 'Features' folder
        - split_perc [int], percentage used to split Features in (split_perc) Train
        			and (1 - split_perc) Test
        - tfrecord: a boolean that indicates to split the .tfrecord file.

        - features_cardinality: the length of the Features TFRecordDataset.

    Outputs (if tfrecord = True):
        - train_cardinality.txt: Text file containing the length of the Train TFRecordDataset.
        - test_cardinality.txt: Text file containing the length of the Test TFRecordDataset.
    '''
    genres = ['Blues', 'Classical', 'Country', 'Disco', 'Hiphop',
              'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']

    if not tfrecord:
        test_path = os.path.join(''.join(path.split(os.sep)[0:-1]), 'Test')
        train_path = os.path.join(''.join(path.split(os.sep)[0:-1]), 'Train')
        for t_path in [test_path, train_path]:
            if os.path.exists(t_path):
                rmtree(t_path)  # delete Test/Train folder if it exists
            os.mkdir(t_path)  # creates Test/Train folder and subfolder for each genre
            [os.mkdir(os.path.join(t_path, folder)) for folder in genres]

        for subdir, dirs, files in os.walk(path):
            if subdir == path:
                continue  # skip first iteration (parent folder), get to subfolders
            nr_of_test_files = len(files) - int(split_perc / 100 * len(files))

            test_files_nr = np.random.choice(np.linspace(0, len(files) - 1, len(files), dtype='int'),
                                             replace=False, size=nr_of_test_files)

            dst_test = os.path.join(test_path, subdir.split(os.sep)[-1])
            dst_train = os.path.join(train_path, subdir.split(os.sep)[-1])
            for file in files:
                if int(file.split('.')[-2]) in test_files_nr:
                    copy(os.path.join(subdir, file), dst_test)
                else:
                    copy(os.path.join(subdir, file), dst_train)
    else:
        if not os.path.exists('..' + os.sep + 'TFRecord'):
            os.mkdir('..' + os.sep + 'TFRecord')  # create the TFrecord folder if it does not exist
        # initialize the writers for the test&train TFRecords
        trainwriter = tf.io.TFRecordWriter('..' + os.sep + 'TFRecord' + os.sep + 'Train_Features.tfrecord')
        testwriter = tf.io.TFRecordWriter('..' + os.sep + 'TFRecord' + os.sep + 'Test_Features.tfrecord')
        # open the feature TFRecord that will be split into test and train
        raw_dataset = tf.data.TFRecordDataset(os.path.join('..', 'TFRecord', 'Features.tfrecord'))
        parsed_dataset = raw_dataset.map(parse_window_function)

        window_counter = 0
        songs_per_genre = len(os.listdir('..' + os.sep + 'GTZAN' + os.sep + 'Blues'))
        no_of_genres = len(os.listdir('..' + os.sep + 'GTZAN'))
        no_of_songs = songs_per_genre * no_of_genres
        windows_per_song = features_cardinality // no_of_songs
        nr_of_test_files = songs_per_genre - int((split_perc / 100) * songs_per_genre)

        train_cardinality, test_cardinality = 0, 0  # will count the train&test window numbers
        windows_per_genre = windows_per_song*songs_per_genre

        for window_data in parsed_dataset:
            window_data['window'] = bytes_feature(window_data['window'])
            window_data['label'] = bytes_feature(window_data['label'] )
            ftfeature = tf.train.Features(feature=window_data)
            ftexample = tf.train.Example(features = ftfeature)
            ftserialized = ftexample.SerializeToString()

            song_counter = window_counter // windows_per_song
            song_in_genre = song_counter % songs_per_genre

            # when the genre changes randomize the songs that will be added to the test dataset from every genre
            if window_counter%windows_per_genre ==0:
                test_files_nr = np.random.choice(np.linspace(0, songs_per_genre - 1, songs_per_genre, dtype='int'),
                                                 replace=False, size=nr_of_test_files)

            if song_in_genre in test_files_nr:
                testwriter.write(ftserialized)
                test_cardinality += 1
            else:
                trainwriter.write(ftserialized)
                train_cardinality += 1
            window_counter += 1

        if not os.path.exists(os.path.join('..', 'Cardinality')):
            os.mkdir(os.path.join('..', 'Cardinality'))
        path_train = os.path.join('..', 'Cardinality', 'Train_Features.txt')
        path_test = os.path.join('..', 'Cardinality', 'Test_Features.txt')
        write_cardinality(path_train, train_cardinality)
        write_cardinality(path_test, test_cardinality)




