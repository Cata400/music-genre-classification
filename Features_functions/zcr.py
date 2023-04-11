def zero_crosssing_rate(self):
    return librosa.feature.zero_crossing_rate(self.signal)