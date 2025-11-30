"""
signal_utils.py - Signal Processing Utilities for VITAL-SENSE rPPG Heart Rate Monitor

This module contains the mathematical functions for processing the raw green channel
signal extracted from the forehead ROI to calculate heart rate.

Core Functions:
- Detrending: Remove slow light variations
- Normalization: Standardize the signal
- Bandpass Filter: Filter frequencies outside 0.7Hz - 4.0Hz (42 BPM - 240 BPM)
- FFT: Convert time-domain signal to frequency-domain and find peak frequency
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, detrend


def detrend_signal(raw_signal: np.ndarray) -> np.ndarray:
    """
    Remove slow light variations (trends) from the raw signal.
    
    Uses scipy's detrend function to remove linear trends from the signal,
    which helps eliminate slow variations caused by ambient lighting changes.
    
    Args:
        raw_signal: 1D numpy array of raw green channel mean values
        
    Returns:
        Detrended signal as a 1D numpy array
    """
    if len(raw_signal) < 2:
        return raw_signal
    return detrend(raw_signal, type='linear')


def normalize_signal(input_signal: np.ndarray) -> np.ndarray:
    """
    Standardize the signal using z-score normalization.
    
    Normalizes the signal to have zero mean and unit variance,
    which helps in comparing signals and improves filter performance.
    
    Args:
        input_signal: 1D numpy array of the signal to normalize
        
    Returns:
        Normalized signal as a 1D numpy array
    """
    if len(input_signal) < 2:
        return input_signal
    
    mean_val = np.mean(input_signal)
    std_val = np.std(input_signal)
    
    if std_val == 0:
        return input_signal - mean_val
    
    return (input_signal - mean_val) / std_val


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4) -> tuple:
    """
    Design a Butterworth bandpass filter.
    
    Args:
        lowcut: Low cutoff frequency in Hz
        highcut: High cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order (default: 4)
        
    Returns:
        Tuple of (b, a) filter coefficients
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Clamp values to valid range (0, 1)
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    
    if low >= high:
        low = high - 0.01
    
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data: np.ndarray, lowcut: float = 0.7, highcut: float = 4.0,
                    fs: float = 30.0, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the signal.
    
    Filters out frequencies outside the physiological range of heart rate:
    - Low cutoff: 0.7 Hz (42 BPM)
    - High cutoff: 4.0 Hz (240 BPM)
    
    Args:
        data: 1D numpy array of the signal to filter
        lowcut: Low cutoff frequency in Hz (default: 0.7 Hz = 42 BPM)
        highcut: High cutoff frequency in Hz (default: 4.0 Hz = 240 BPM)
        fs: Sampling frequency in Hz (default: 30.0 Hz)
        order: Filter order (default: 4)
        
    Returns:
        Filtered signal as a 1D numpy array
    """
    if len(data) < 13:  # Minimum length for filtfilt with default padlen
        return data
    
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    
    # Use filtfilt for zero-phase filtering
    try:
        filtered_signal = filtfilt(b, a, data)
    except ValueError:
        # If filter fails, return original data
        return data
    
    return filtered_signal


def compute_fft(data: np.ndarray, fs: float = 30.0) -> tuple:
    """
    Compute the Fast Fourier Transform of the signal.
    
    Converts the time-domain signal to frequency-domain to identify
    the dominant frequency (heart rate).
    
    Args:
        data: 1D numpy array of the filtered signal
        fs: Sampling frequency in Hz (default: 30.0 Hz)
        
    Returns:
        Tuple of (frequencies, magnitudes) arrays
    """
    n = len(data)
    if n < 2:
        return np.array([0]), np.array([0])
    
    # Compute FFT
    fft_result = np.fft.rfft(data)
    magnitudes = np.abs(fft_result)
    
    # Compute frequency bins
    frequencies = np.fft.rfftfreq(n, d=1.0/fs)
    
    return frequencies, magnitudes


def find_peak_frequency(frequencies: np.ndarray, magnitudes: np.ndarray,
                        min_freq: float = 0.7, max_freq: float = 4.0) -> float:
    """
    Find the dominant frequency (peak) in the specified range.
    
    Args:
        frequencies: Array of frequency values
        magnitudes: Array of magnitude values
        min_freq: Minimum frequency to consider (default: 0.7 Hz)
        max_freq: Maximum frequency to consider (default: 4.0 Hz)
        
    Returns:
        Peak frequency in Hz
    """
    if len(frequencies) < 2 or len(magnitudes) < 2:
        return 0.0
    
    # Find indices within the valid frequency range
    valid_indices = np.where((frequencies >= min_freq) & (frequencies <= max_freq))[0]
    
    if len(valid_indices) == 0:
        return 0.0
    
    # Find the peak within valid range
    valid_magnitudes = magnitudes[valid_indices]
    peak_idx = valid_indices[np.argmax(valid_magnitudes)]
    
    return frequencies[peak_idx]


def frequency_to_bpm(frequency: float) -> float:
    """
    Convert frequency in Hz to BPM (beats per minute).
    
    Args:
        frequency: Heart rate frequency in Hz
        
    Returns:
        Heart rate in BPM
    """
    return frequency * 60.0


def process_signal(raw_signal: np.ndarray, fs: float = 30.0) -> tuple:
    """
    Complete signal processing pipeline for rPPG.
    
    Applies the full processing chain:
    1. Detrending - Remove slow light variations
    2. Normalization - Standardize the signal
    3. Bandpass Filtering - Filter out non-physiological frequencies
    4. FFT - Find the dominant frequency
    5. BPM Calculation - Convert frequency to heart rate
    
    Args:
        raw_signal: 1D numpy array of raw green channel mean values
        fs: Sampling frequency in Hz (default: 30.0 Hz)
        
    Returns:
        Tuple of (bpm, filtered_signal)
        - bpm: Calculated heart rate in beats per minute
        - filtered_signal: The processed signal for visualization
    """
    if len(raw_signal) < 30:  # Need at least 1 second of data at 30 fps
        return 0.0, raw_signal
    
    # Step 1: Detrend the signal
    detrended = detrend_signal(raw_signal)
    
    # Step 2: Normalize the signal
    normalized = normalize_signal(detrended)
    
    # Step 3: Apply bandpass filter
    filtered = bandpass_filter(normalized, lowcut=0.7, highcut=4.0, fs=fs)
    
    # Step 4: Compute FFT
    frequencies, magnitudes = compute_fft(filtered, fs)
    
    # Step 5: Find peak frequency and convert to BPM
    peak_freq = find_peak_frequency(frequencies, magnitudes)
    bpm = frequency_to_bpm(peak_freq)
    
    return bpm, filtered


def calculate_signal_quality(filtered_signal: np.ndarray) -> float:
    """
    Calculate the quality/stability of the signal.
    
    Uses the coefficient of variation to estimate signal quality.
    A stable signal will have consistent peaks and lower variance.
    
    Args:
        filtered_signal: The processed/filtered signal
        
    Returns:
        Quality score from 0.0 (poor) to 1.0 (excellent)
    """
    if len(filtered_signal) < 10:
        return 0.0
    
    # Calculate coefficient of variation
    std_val = np.std(filtered_signal)
    mean_val = np.mean(np.abs(filtered_signal))
    
    if mean_val == 0:
        return 0.0
    
    cv = std_val / mean_val
    
    # Convert to quality score (lower CV = higher quality)
    # Typical good signal has CV around 0.5-2.0
    quality = max(0.0, min(1.0, 1.0 - (cv / 5.0)))
    
    return quality
