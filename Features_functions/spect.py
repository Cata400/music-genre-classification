def spectrogram(self):
    spect = amplitude(librosa.stft(self.signal, n_fft=self.n_fft, hop_length=self.hop_length))
    setattr(self, 'spect', spect)
    return spect
