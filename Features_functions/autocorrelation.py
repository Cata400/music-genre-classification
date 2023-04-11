def autocor(self):
    autocorr = autocorrelation(self.signal, self.n_fft)
    return autocorr[:, 0:autocorr.shape[1] // 2 + 1]