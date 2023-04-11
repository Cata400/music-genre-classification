def chroma_stft(self):
    return librosa.feature.chroma_stft(S=self.spect ** 2, sr=self.sr)