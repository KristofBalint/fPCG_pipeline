import pyPCG
import scipy.io.wavfile as wav
from scipy import signal
import scipy.fft as fft
import numpy as np

def segment_audio(fajlnev, fs=4000, segment_length=30, step_size=5):
    '''
    This code makes overlapping audio segments, to augment the number of recordings
    :param fajlnev: name of the file you read in
    :param fs: sample frequency
    :param segment_length: The lenght of the returned segments in seconds
    :param step_size: the offset of the segment, in seconds.
    This gives the step_size*n times of the starting point of the segments.
    :return: overlapping segments
    '''
    audio, _ = pyPCG.io.read_signal_file(fajlnev, "wav")  # Load sound
    segment_samples = segment_length * fs  # Convert to samples
    step_samples = step_size * fs  # Convert step size to samples

    segments = []
    for start in range(0, len(audio) - segment_samples + 1, step_samples):
        segments.append(audio[start : start + segment_samples])  # Slice segment

    return segments

def pair_sound_data(records,path,maternal_data):
    '''
    This code pairs the records with the metadata, such as maternal age,
    pregnancy term, BMI, fetal sex, clinical background
    :param records: PCG audio recordings
    :param path: what folder should the code search the records in
    :param maternal_data: a pandas dataframe containing metadata
    :return: an array with all the recordings paired with metadata
    '''
    sound_with_data = []
    for filename in records:
        sound_file = path + filename + ".wav"
        origrate, data = wav.read(sound_file)
        ratio = 5000 / origrate
        downsampled_data = signal.resample(data, int(len(data) * ratio))
        # sound_data, fs=pyPCG.io.read_signal_file(downsampled_sound_file,"wav")
        # sound=pyPCG.pcg_signal(*pyPCG.io.read_signal_file(sound_file,"wav"))
        sound = pyPCG.pcg_signal(downsampled_data[:, 0], 5000)

        only_number = filename.replace(".wav", "")
        only_number = only_number.split("-")[0]
        only_number = only_number[1:]
        # print(maternal_data["Pregnancy Term (weeks)"].where(maternal_data["Subject ID"].str.endswith(only_number)))
        maternal_data_clean = maternal_data.dropna(subset=["Subject ID"])
        matched_term = maternal_data_clean.loc[
            maternal_data_clean["Subject ID"].str.endswith(
                only_number), "Pregnancy Term (weeks)"]
        if not matched_term.empty:
            pregnancy_term = matched_term.iloc[0]  # Take the first match
            print(f"Pregnancy Term for {filename}: {pregnancy_term}")
        else:
            print(f"No match found for term in {filename}")
        matched_BMI = maternal_data_clean.loc[
            maternal_data_clean["Subject ID"].str.endswith(
                only_number), "Maternal BMI"]
        if not matched_BMI.empty:
            pregnancy_BMI = matched_BMI.iloc[0]
        else:
            print(f"No match found for BMI in {filename}")
        matched_age = maternal_data_clean.loc[
            maternal_data_clean["Subject ID"].str.endswith(
                only_number), "Mother’s age (years)"]
        if not matched_age.empty:
            pregnancy_age = matched_age.iloc[0]
        else:
            print(f"No match found for mother's age in {filename}")
        clin_hist = maternal_data_clean.loc[
            maternal_data_clean["Subject ID"].str.endswith(
                only_number), "Clinical History"]
        if not clin_hist.empty:
            hist = clin_hist.iloc[0]
        else:
            hist = "no"

        matched_gender = maternal_data_clean.loc[
            maternal_data_clean["Subject ID"].str.endswith(
                only_number), "Fetus gender (B: Boy, G: Girl)"]
        if not matched_gender.empty:
            baby_gender = matched_gender.iloc[0]
        else:
            print(f"No match found for gender in {filename}")

        cutting = segment_audio(sound_file, 5000, 30, 5)
        # print(len(sound),fs)
        if filename[0] == "f":
            sound_with_data.append((sound, pregnancy_term, pregnancy_BMI,
                                    pregnancy_age, hist, baby_gender, filename))
            for cut in cutting:
                sound = pyPCG.pcg_signal(cut[:, 0], 5000)
                sound_with_data.append((sound, pregnancy_term, pregnancy_BMI,
                                        pregnancy_age, hist, baby_gender,
                                        filename))
    return sound_with_data

def spectral_spread_optimized(start, end, signal_data, factor=0.7, nfft=512):
    '''
    Calculate spectral spread of the segments, percentage of the total power
    of the segment and the frequency difference between the beginning
    and end of the calculated area
    :param start: start time of the recording
    :param end: end time of the recording
    :param signal_data: input signal
    :param factor: percentage of total power. Defaults to 0.7.
    :param nfft: resolution of the Fourier transform
    :return: difference of the beginning and end of the given area as an np.ndarray
    '''
    start, end = np.asarray(start), np.asarray(end)
    num_segments = len(start)
    ret = np.zeros(num_segments)

    for i in range(num_segments):
        s, e = start[i], end[i]
        segment = signal_data[s:e]
        spect = np.abs(fft.fft(segment, n=nfft))[:nfft//2]
        power = spect**2
        cumulative_power = np.cumsum(np.sort(power)[::-1])

        threshold = np.sum(power) * factor
        min_index = np.searchsorted(cumulative_power, threshold)

        peak = np.argmax(power)
        ret[i] = abs(peak - min_index)

    return ret

import numpy as np
import numpy.typing as npt

def peak_spread_optimized(
    start: npt.NDArray[np.int_],
    end: npt.NDArray[np.int_],
    envelope_data: npt.NDArray[np.float64],  # Pass the data array directly
    fs: float,  # Pass the sampling rate
    factor: float = 0.7
) -> npt.NDArray[np.int_]:
    """Optimized peak spread calculation.

    Args:
        start (np.ndarray): start times in samples
        end (np.ndarray): end times in samples
        envelope_data (np.ndarray): The data array from the envelope signal
        fs (float): The sampling rate of the envelope signal
        factor (float, optional): percentage of total area. Defaults to 0.7.

    Returns:
        np.ndarray: time difference between the beginning and end of the percentage area in ms
    """
    ret = np.zeros(len(start), dtype=np.float64)

    for i in range(len(start)):
        s, e = start[i], end[i]
        win = envelope_data[s:e]  # Use the data array
        total_area = np.sum(win) * factor

        peak = np.argmax(win)
        power_cumsum = np.cumsum(win)

        left_idx = np.searchsorted(power_cumsum, power_cumsum[peak] - total_area, side="left")
        right_idx = np.searchsorted(power_cumsum, power_cumsum[peak] + total_area, side="right")

        spread = right_idx - left_idx
        ret[i] = spread / fs * 1000  # Use the passed fs

    return ret

def categorize_pregnancy_term(term):
    '''
    Categorize pregnancy term, based on the median, q1, q3 ranges.
    :param term: a number representing the week of the pregnancy
    :return: a string representing categories
    '''
    if term <= 35:
        return "35-nél korábbi"
    elif 36 <= term <= 37:
        return "36-37"
    elif 38 <= term <= 39:
        return "38-39"
    else:
        return "40-nél későbbi"

def categorize_bmi(bmi):
    '''
    Categorize BMI, based on the median, q1, q3 ranges.
    :param bmi: a number representing body mass index
    :return: a string representing categories
    '''
    if bmi<=26.275:
        return "26-nál kisebb"
    elif 26.275<=bmi<=28.65:
        return "26-28"
    elif 28.65<=bmi<=31.6:
        return "28-31"
    else:
        return "31-nél nagyobb"

def categorize_age(age):
    '''
    Categorize age, based on the median, q1, q3 ranges.
    :param age: age of the mother in years
    :return: a string representing categories
    '''
    if age<=25:
        return "25-nál fiatalabb"
    elif 25<=age<=29:
        return "25-29"
    elif 29<=age<=33:
        return "25-33"
    else:
        return "33-nál idősebb"
