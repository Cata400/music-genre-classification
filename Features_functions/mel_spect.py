def mel_spectrogram(self):
    mel_spect = librosa.feature.melspectrogram(S=self.spect ** 2, sr=self.sr)
    setattr(self, 'mel_spect', mel_spect)
    return mel_spect
