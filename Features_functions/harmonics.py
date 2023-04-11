def harmonic_elements(self):
    return librosa.effects.harmonic(self.signal, margin=self.margin)
