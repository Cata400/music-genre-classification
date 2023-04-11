def spectral_rolloff(self):
    return librosa.feature.spectral_rolloff(S=self.spect, sr=self.sr)