from utils import *
from iterate_dataset import iterate_dataset
from split_dataset import split_dataset
from k_fold_cross_validation import k_fold_cross_validation
from feature_extractor import *



def main():
    iterate = False  #[boolean], extract features by parsing the dataset
    split = True  #[boolean], split the data according to split_perc
    tfrecord = True  #[boolean], save features in a TFRecord file
    train = True  #[boolean], train a new model, otherwise use the model from path_model
    path_model = os.path.join('..', 'Model', 'model_1.h5')  #[path], relative path to save/load model to/from
    predict = True  #[boolean], predict using a model that is loaded from path_model


    # Arguments for iterate dataset
    gtzan_path = os.path.join('..', 'GTZAN')  #[path], relative path to GTZAN database
    window_length = 3 * 22050  #[int], length of a window in samples (1s = 22050 samples)
    window_type = 'rect'  #[string], type of window e.g: rect, blackman, etc
    overlap = 0  #[int] window overlap percentage (between 0 and 100)
    feature_extractor = FeatureExtractor()
    feature_list = ['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth', 
                    'spectral_rolloff', 'zcr', 'harmonics', 'tempo', 'mfcc']  #[list] of features to be extracted, see feature_extractor.py for details
    feature_dict = {'sr': 22050, 'n_fft': 2048, 'n_mfcc': 13, 'hop_length': 512, 'margin': 3.0}  #[dict] of args for feature extractor
    window = True  # [boolean], True = the features will be extracted window by window, False = directly from the array of signals.
    normalize = False  #[boolean], normalize the signal (song) in range [-1;1]
    standardize = False  #[bool], standardize the signal (song). For each sample x: x = (x - mean(signal)) / std(signal)
    variance_type = 'smad'  #[string], type of variance, either 'var' or 'smad'

    # Arguments for split dataset
    split_path = os.path.join('..', 'Features')  #[path], relative path to Features folder
    split_perc = 90  #[int], percentage of Train vs Test split (between 0 and 100)

    # Arguments for k_fold_cross_validation (train)
    k_fold_path = os.path.join('..', 'Train')  #[path], relative path to Train folder
    k = 10  #[int], number of folds to be performed
    batch_size = 1024  #[int], size of batch in examples (windows)
    shuffle_buffer = 3 * batch_size  #[int], size of the buffer used to shuffle the data
    epochs = 250  #[int], number of epochs to be performed during training
    optimizer = 'adam'  #[string or tensorflow.keras.optimizers], optimizer to be used
    dropout = 0.5  #[float], between 0 and 1. Fraction of the input units to drop
    shuffle_mode = True #[string], if True shuffles train and validation datasets as one dataset, else individualy

    # Arguments for predict (test)
    per_class_accuracy = True  #[boolean], prints the accuracy of each class
    confusion_matrix = True  #[boolean], plots the confusion matrix
    roc_curve = True  #[boolean], plots the ROC curve with Area Under Curve calculated


    # Start of program
    if iterate:
        iterate_dataset(gtzan_path, window_length, window_type, overlap, feature_extractor, feature_dict, feature_list,
                        window, normalize, standardize, tfrecord, variance_type)
        
    if split:
        if tfrecord:
            features_cardinality = load_cardinality(os.path.join('..', 'Cardinality', 'Features.txt'))
        else:
            features_cardinality = None
        split_dataset(split_path, split_perc, features_cardinality, tfrecord)

    scaller = compute_scaller(tfrecord)

    if train:
        if tfrecord:
            train_cardinality = load_cardinality(os.path.join('..', 'Cardinality', 'Train_Features.txt'))
        else:
            train_cardinality = None
        k_fold_cross_validation(k_fold_path, k, path_model, tfrecord, train_cardinality, batch_size, shuffle_buffer,
                                      epochs, optimizer, dropout, scaller, shuffle_mode)

    if predict:
        model = tf.keras.models.load_model(path_model)
        if tfrecord:
            x_test, y_true = create_test_tfrecord(os.path.join('..', 'TFRecord', 'Test_Features.tfrecord'), scaller)
        else:
            x_test, y_true = create_test_npy(os.path.join('..', 'Test'), window_length, scaller)
        y_score = test_model(model, x_test)
        y_pred = np.argmax(y_score, axis=1)
        if confusion_matrix:
            plot_confusion_matrix(y_true, y_pred)
        if per_class_accuracy:
            get_per_class_accuracy(y_true, y_pred)
        if roc_curve:
            plot_roc_curve(y_true, y_score)
    
if __name__ == '__main__':
    main()
