from utils import *
from iterate_dataset import *


def test_iterate_dataset(path, window_length, window_type, overlap, feature_type, rec_mode, tfrecord=True, *args):
    """
    Python script to test the iterate_dataset function

    The script prints the MSE and generates 2 graphs:
        1. The original signal and the reconstructed one
        2. Mean Squared Error in each point

     Arguments:
         -path: the relative path to the GTZAN dataset
         -window_length: length of a signal window (in samples)
         -window_type: the type of window used to split the signal, e.g. 'blackman', 'hamming', 'hanning', etc.
         -overlap: the percentage of overlapping between windows
         -feature_type: the feature to be extracted from a signal window
         -rec_mode [string], reconstruction mode, usually 'OLA' or 'MEAN' check utils.sigrec() for more details
         -tfrecord: a boolean that indicates whether the .tfrecord conversion should apply.
         -*args: any necessary arguments for the required feature:
                 - for 'autocorrelation': N_fft[int], no.of points to perform FFT
                 - for 'cepstrum': N_fft[int]
                 - for 'DWT': wavelet_type[string], check DWT function in utils for details
                 - for 'FFT': N_ftt[int]

    """

    if not os.path.exists(path):
        raise Exception("Specified path does not exist!")

    iterate_dataset(path, window_length, window_type, overlap, feature_type, True, False, tfrecord, *args)

    if not tfrecord:
        for genre in os.listdir("../Features"):
            count = 0
            for file in os.listdir(os.path.join("../Features", genre)):
                count += 1
            if count != 100:
                print("The " + genre + " folder is missing " + str(100 - count) + " feature files")
            else:
                print("The " + genre + " folder contains all the files")


    # Testing on the rock.00099.wav file from the GTZAN database
    # Signal reconstruction

    sample_rate, song = wavfile.read(os.path.join(path, "Rock", "rock.00099.wav"))
    song = convert_to_sg_ch(song)
    song = normalization(song)
    w_signal = sigwin(song, window_length, window_type, overlap)

    if feature_type == 'FFT':
        if not tfrecord:
            extracted = np.load(os.path.join("../Features", "Rock", "rock.00099.npy"))
            extracted = extracted[:, 0:len(extracted[0]) - 1]
        else:
            card_path = os.path.join("..", "Cardinality", "Features.txt")
            features_cardinality = load_cardinality(card_path)

            extracted = []
            raw_dataset = tf.data.TFRecordDataset(os.path.join('..', 'TFRecord', 'Features.tfrecord'))
            decoded_map_dataset = raw_dataset.map(parse_and_decode_function)


            songs_per_genre = len(os.listdir('..' + os.sep + 'GTZAN' + os.sep + 'Blues'))
            no_of_genres = len(os.listdir('..' + os.sep + 'GTZAN'))
            no_of_songs = songs_per_genre * no_of_genres
            windows_per_song = features_cardinality // no_of_songs

            start_index = windows_per_song * (no_of_songs-1)
            wanted_start = decoded_map_dataset.skip(start_index)
            wanted_song = wanted_start.take(windows_per_song).as_numpy_iterator()
            for window_data in wanted_song:
                extracted.append(window_data[0][0:-1])
        extracted = np.asanyarray(extracted)
        fft_amplitude = np.concatenate((extracted, np.flip(extracted, axis=1)), axis=1)
        fft_phase = phase(FFT(w_signal, args[0]))
        fft = np.multiply(fft_amplitude, np.exp(fft_phase * 1j))
        ifft = IFFT(fft, window_length)
        rec = sigrec(ifft.real, overlap, rec_mode)

        # Plots
        fig1 = plt.figure()
        axes1 = fig1.add_axes([0.13, 0.1, 0.85, 0.67])
        mod = 'Overlap-add' if rec_mode == 'OLA' else 'Overlap-mean'
        text = mod + ' method is used at reconstruction\n'
        time = np.linspace(0, np.shape(rec)[0] // sample_rate, np.shape(rec)[0])

        axes1.plot(time, song[0:np.shape(rec)[0]], 'blue')
        axes1.plot(time, rec, 'red')
        axes1.set_xlabel('time [seconds]')
        axes1.set_ylabel('Amplitude (normed)')
        axes1.set_title('Original song vs Reconstructed song' + '\nfor window type: ' +
                        str(window_type) + ', overlap: ' + str(overlap) + '%\n and window length: ' +
                        format(1000 * window_length / sample_rate, '.2f') + ' ms\n' + text)
        axes1.legend(('Original', 'Reconstructed'))
        plt.show()

        fig2 = plt.figure()
        axes2 = fig2.add_axes([0.13, 0.1, 0.75, 0.72])
        mse, mse_points = MSE(song, rec)
        axes2.plot(time, mse_points)
        axes2.set_xlabel('time [seconds]')
        axes2.set_ylabel('MSE value')
        axes2.set_title('Mean squared error\n' + 'For window type: ' + str(window_type) +
                        ', overlap: ' + str(overlap) + '% and window length: ' +
                        format(1000 * window_length / sample_rate, '.2f') + ' ms\n' + text)
        plt.show()

        print('MSE value is: ', mse)

    return rec


NPY = test_iterate_dataset(os.path.join('..', 'GTZAN'), 22050, "blackman", 60, "FFT", 'OLA', False, next_pow_of_2(22050))
TFR = test_iterate_dataset(os.path.join('..', 'GTZAN'), 22050, "blackman", 60, "FFT", 'OLA', True, next_pow_of_2(22050))
print(np.allclose(NPY, TFR))