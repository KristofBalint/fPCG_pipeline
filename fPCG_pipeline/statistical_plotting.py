import matplotlib.pyplot as plt
import seaborn as sns
from .preprocessing import categorize_pregnancy_term, categorize_bmi, categorize_age
import numpy as np
from scipy import stats
import scikit_posthocs as sp
from matplotlib.transforms import Transform
from matplotlib.scale import ScaleBase
from matplotlib import scale as mscale
import scipy
import pyPCG.preprocessing

def remove_overlap(features_df, olap_rate):
    '''

    :param features_df:
    :param olap_rate: segment_lenght/step_size from segment_audio
    :return: a dataframe where each row corresponds to a non-overlapping segment
    '''
    # start_index = features_df[features_df["Pregnancy term category"] != '35-nél korábbi'].index[0]
    num_rows = int((len(features_df)) / olap_rate)
    indices_to_select = [0 + i * int(olap_rate) for i in range(num_rows)]
    non_overlapping = features_df.iloc[indices_to_select]
    non_overlapping = non_overlapping.reset_index(drop=True)
    return non_overlapping

def apply_category(non_overlapping):
    '''

    :param non_overlapping: input dataframe
    :return: dataframe with categorized metadata
    '''
    non_overlapping['Pregnancy term category'] = non_overlapping[
        'Pregnancy term category'].apply(categorize_pregnancy_term)
    non_overlapping['BMI category'] = non_overlapping['BMI category'].apply(
        categorize_bmi)
    non_overlapping['Age category'] = non_overlapping['Age category'].apply(
        categorize_age)


def plot_correlation_matrix(non_overlapping,column_renames,feats_to_drop, thresh=0.3
                            ,fmt=".1f",ttl_fsize=20,tck_fsize=16,xrot=45,yrot=15):
    '''
    :param tck_fsize: tick size
    :param ttl_fsize: title font size
    :param column_renames: Renaming columns in order to reduce the size of the figure
    :param non_overlapping: input dataframe
    :param feats_to_drop: features not visualised in the matrix
    :param thresh: only features above the threshold will be visible
    :param fmt: how many digits of the correlation score should be displayed
    :return: plots a correlation matrix
    '''


    corr_matrix = non_overlapping.copy().rename(columns=column_renames)

    corr_matrix = corr_matrix.drop(
        feats_to_drop, axis=1)

    corr_matrix = corr_matrix.corr()
    corr_matrix = corr_matrix[abs(corr_matrix) >= thresh]
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=fmt)
    plt.title('Correlation Matrix', fontsize=ttl_fsize)
    plt.xticks(rotation=xrot, ha='right',fontsize=tck_fsize)  # Rotate x-axis labels
    plt.yticks(rotation=yrot, ha='right',fontsize=tck_fsize)
    plt.show()

def plot_viol(non_overlapping, X , ys,x_rename,lbl_fsize=18,tck_fsize=12):
    '''
    Modular implementation of making violin plots for features.
    :param x_rename: renaming the data from 0 1 to something more understandable
    :param non_overlapping: input dataframe
    :param X: the variable we want to compare the distributions of the classes
    :param ys: for what features we compare
    :return:plots violins, and prints results of Kruskal-Wallis and Dunn post-hoc tests
    '''
    plt.figure(figsize=(7*len(ys), 8))
    groups=non_overlapping[X].unique()
    for i in range(len(ys)):
        plt.subplot(1, len(ys), i+1)
        sns.violinplot(data=non_overlapping,x=X,y=ys[i])
        plt.xlabel(X, fontsize=lbl_fsize)
        plt.ylabel(ys[i], fontsize=lbl_fsize)
        plt.xticks(x_rename[0],x_rename[1],fontsize=tck_fsize)
        plt.yticks(fontsize=tck_fsize)
        data = [non_overlapping[ys[i]][non_overlapping[X] == g]  for g in groups]
        H, p = stats.kruskal(*data)
        print(
            f"Kruskal-Wallis H test for {ys[i]} rate: H = {H:.2f}, p-value = {p:.3f}")
        dunn_result = sp.posthoc_dunn(data, p_adjust='bonferroni')
        print(dunn_result)

def pairplot(non_overlapping,feat_list,hue):
    '''

    :param non_overlapping: input dataframe
    :param feat_list: features visualised in the pairplot
    :param hue: the feature that will be highlighted
    :return: plots a pairplot
    '''
    sns.pairplot(non_overlapping[feat_list], hue=hue)


class MelTransform(Transform):
    """
    Transforms frequency(Hz) to mel scale.
    """
    input_dims = 1
    output_dims = 1
    is_separable = True

    def transform_non_affine(self, a):
        return 2595 * np.log10(1 + a / 700.0)

    def inverted(self):
        return InverseMelTransform()

class InverseMelTransform(Transform):
    """
    Inverts Mel scale back to frequency (Hz).
    """
    input_dims = 1
    output_dims = 1
    is_separable = True

    def transform_non_affine(self, a):

        return 700.0 * (10**(a / 2595.0) - 1)

    def inverted(self):
        return MelTransform()


class MelScale(ScaleBase):
    """
    Matplotlib scale class for Mel scale.
    """
    name = 'mel' # This name is used in set_yscale('mel')

    def __init__(self, axis, **kwargs):
        ScaleBase.__init__(self, axis)

    def get_transform(self):
        return MelTransform()

    def set_default_locators_and_formatters(self, axis):
        pass # The default locators are not good for mel scale

def spectrogram_and_raw_signal(data,start_time,end_time,fs, NFFT_val = 2048,
                               ymin_freq = 1, ymax_freq = 8000,
                               vmin_dB = -30, vmax_dB = 50,cmap='jet'  ):
    '''
    Displays a segment of a raw signal and its spectrogram
    :param data: sound data
    :param start_time: starting time of the plot
    :param end_time: ending time of the plot
    :param fs: frequency
    :param NFFT_val: N for the fast Fourier transfrom
    :param ymin_freq: minimum freqency for the displayed spectrogram
    :param ymax_freq: maximum freqency for the displayed spectrogram
    :param vmin_dB: minimum dB for the displayed spectrogram
    :param vmax_dB: maximum dB for the displayed spectrogram
    :param cmap: The colormap for the spectrogram
    :return:
    '''
    mscale.register_scale(MelScale)
    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)
    sliced_sound_data = data[start_sample:end_sample, 0]

    fig, axes = plt.subplots(2, 1, figsize=(20, 12),
                             gridspec_kw={'height_ratios': [1, 1]})


    window_val = NFFT_val
    noverlap_val = NFFT_val // 2048

    time_axis_seconds = np.linspace(start_time, end_time,
                                    len(sliced_sound_data), endpoint=False)
    im = axes[0].specgram(
        sliced_sound_data,
        Fs=fs,
        NFFT=NFFT_val,
        noverlap=noverlap_val,
        window=np.hanning(window_val),  # Hann window
        scale='dB',  # Intensity scale in dB
        vmin=vmin_dB,
        vmax=vmax_dB,
        cmap=cmap,
    )
    # axes[0].set_xlim(start_time, end_time)
    axes[0].set_yscale('mel')
    axes[0].set_ylim(ymin=ymin_freq, ymax=ymax_freq)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].set_xlabel("Time (s)", fontsize=25)
    axes[0].set_ylabel("Frequency (Hz)", fontsize=25)
    axes[0].set_title(
        f"Spectrogram of fPCG record ({start_time}-{end_time} seconds)",
        fontsize=30)



    # Plot the sliced sound data on the second subplot

    axes[1].plot(time_axis_seconds, sliced_sound_data)
    axes[1].set_xlim(start_time, end_time)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    axes[1].set_xlabel("Time (seconds)", fontsize=25)
    axes[1].set_ylabel("Amplitude (n.u.)", fontsize=25)
    axes[1].set_title("Raw signal", fontsize=30)

    plt.tight_layout()
    plt.show()

def plot_freq_spectrogram(data,filtered=False):
    '''
    Plots the spectrogram of the raw signal
    :param data: raw signal
    :param filtered: if it needs filtering
    :return:
    '''
    if filtered:
        denoised=pyPCG.preprocessing.wt_denoise_sth(data)
        filtered=pyPCG.preprocessing.filter(denoised,filt_ord=4,filt_cutfreq=350,filt_type='LP')
        data=pyPCG.preprocessing.filter(filtered,filt_ord=4,filt_cutfreq=100,filt_type='HP')
        fs = filtered.fs
    else:
        fs=data.fs
    # Perform Fast Fourier Transform (FFT)
    fft_result = scipy.fft.fft(data.data)
    frequencies = scipy.fft.fftfreq(len(data.data), 1 / fs)

    # Take the absolute value of the FFT result and consider only the positive frequencies
    fft_magnitude = np.abs(fft_result)[:len(fft_result) // 2]
    positive_frequencies = frequencies[:len(frequencies) // 2]

    # Plot the frequency spectrum
    plt.figure(figsize=(20, 6))
    plt.plot(positive_frequencies, fft_magnitude)
    plt.xlabel("Frequency (Hz)", fontsize=20)
    plt.ylabel("Magnitude", fontsize=20)
    plt.title("Frequency Spectrum of Raw fPCG Record", fontsize=20)
    plt.grid(True)
    plt.show()