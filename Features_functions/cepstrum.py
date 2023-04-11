def cepstrum(self):
    C = cepstrum(self.signal, self.n_fft)
    return C[:, 0:C.shape[1] // 2 + 1]