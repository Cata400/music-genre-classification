def fft(self):
    fourier = FFT(self.signal, self.n_fft)
    return amplitude(fourier[:, 0:fourier.shape[1] // 2 + 1])