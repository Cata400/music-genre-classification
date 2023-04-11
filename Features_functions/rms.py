def RMS(self):
    return librosa.feature.rms(y=self.signal)