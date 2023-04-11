def spectral_bandwidth(self):
    return librosa.feature.spectral_bandwidth(S=self.spect, sr=self.sr)
