class Aug:
    """Any augmentation, base"""

class BatchAug(Aug):
    """This augmentation may be apply for batch"""

class WaveAug(Aug):
    """This augmentation can be apply only for waveform based audio"""

    def __str__(self):
        return 'WaveAug'

    def __repr__(self):
        return self.__str__()
    
class SpecAug(Aug):
    """This augmentation can be apply only for spectrogram based audio"""

    def __str__(self):
        return 'SpecAug'

    def __repr__(self):
        return self.__str__()