import numpy as np
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5) -> tuple:
    """Create a bandpass filter using the butterworth method
    たぶんバンドパスフィルタを作成する関数
    """
    return butter(order, [lowcut, highcut], fs=fs, btype="band")


def butter_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5
) -> np.ndarray:
    """Filter a noisy signal
    指定したバンドパスフィルタでノイズを除去する関数

    Args:
        data (np.ndarray): Noisy signal
        lowcut (float): Lower cutoff frequency
        highcut (float): Higher cutoff frequency
        fs (float): Sample rate
        order (int, optional): Order of the filter. Defaults to 5.

    Returns:
        np.ndarray: Filtered signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def denoise_filter(x: np.ndarray) -> np.ndarray:
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 200.0
    lowcut = 1.0
    highcut = 25.0

    # Filter a noisy signal.
    # 200Hzのサンプリング周波数で、1Hzから25Hzの周波数を通すバンドパスフィルタを作成
    # データ列50sec分を対象にフィルタリング
    # ↓３行使ってない？
    # T = 50
    # nsamples = T * fs
    # t = np.arange(0, nsamples) / fs # 使ってなさそう？
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    y = (y + np.roll(y, -1) + np.roll(y, -2) + np.roll(y, -3)) / 4
    y = y[0:-1:4]

    return y
