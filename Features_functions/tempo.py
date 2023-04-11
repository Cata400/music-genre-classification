def tempo(self):
    onset_env = librosa.onset.onset_strength(self.signal, sr=self.sr)
    return librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr)
