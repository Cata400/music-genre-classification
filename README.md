# Music Genre Classification

The purpose of this repository is to classify a melody by its genre using different temporal (such as Root Mean Square, Tempo, Zero Crossing Rate) and spectral (Spectrogram, Mel Frequency Cepstrum Coefficients, Discrete Wavelet Transform, etc) features. The Neural Network was trained on the GTZAN Dataset, which contains 1000 songs, equally divided into 10 genres, using the K-Fold Cross Validation technique. An accuracy of 80% was achieved, while the per-class accuracies range from 63% for Disco to 93% for Classical Music.

The directory tree of the project should be:
<pre>
├── music_genre_classification
├── GTZAN
├── Features
├── Test
└── Train
</pre>
