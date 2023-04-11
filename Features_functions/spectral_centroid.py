def spectral_centroid(self):
    return librosa.feature.spectral_centroid(S=self.spect, sr=self.sr)