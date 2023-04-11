def mfcc(self):
    return librosa.feature.mfcc(S=librosa.power_to_db(self.mel_spect), n_mfcc=self.n_mfcc)
