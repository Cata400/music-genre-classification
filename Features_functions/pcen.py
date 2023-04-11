def pcen(self):
    return librosa.pcen(S=self.spect, sr=self.sr, hop_length=self.hop_length)
