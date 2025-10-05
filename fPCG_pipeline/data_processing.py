import pyPCG
import numpy as np
import time
from preprocessing import *
import pandas as pd

def process(sound_with_data, categorize=False, save=True):
    '''
    Goes through the dataset, calculating all pyPCG features, and puts them
    in a dataframe, and even saves it in a .csv file if needed.
    :param sound_with_data: All the recordings paired with metadata
    :param categorize: a boolean indicating whether to categorize the recordings
    :param save: a boolean indicating whether to save the dataframe
    :return: a dataframe containing metadata, statistics and calculated features
    '''

    features_list = []

    for i in range(len(sound_with_data)):
        filename = sound_with_data[i][0]
        pregnancy_term = sound_with_data[i][1]
        BMI = sound_with_data[i][2]
        age = sound_with_data[i][3]
        clin_hist = sound_with_data[i][4]
        baby_gender = sound_with_data[i][5]
        denoised = pyPCG.preprocessing.wt_denoise_sth(filename)
        filtered = pyPCG.preprocessing.filter(denoised, filt_ord=4,
                                              filt_cutfreq=350, filt_type='LP')
        filtered = pyPCG.preprocessing.filter(filtered, filt_ord=4,
                                              filt_cutfreq=100, filt_type='HP')
        envelope = pyPCG.preprocessing.envelope(filtered)

        # segmentation
        segment_startend = pyPCG.segment.segment_peaks(
            pyPCG.segment.adv_peak(filtered)[1], filtered)
        startings = np.sort(segment_startend[0])
        endings = np.sort(segment_startend[1])

        # statistics
        maxim = pyPCG.stats.max(filtered.data)
        minim = pyPCG.stats.min(filtered.data)
        mean = pyPCG.stats.mean(filtered.data)
        std = pyPCG.stats.std(filtered.data)
        median = pyPCG.stats.med(filtered.data)

        # features
        time_1 = time.perf_counter()
        t_delt = pyPCG.features.time_delta(startings, endings, filtered)
        time_2 = time.perf_counter()
        zero_cr = pyPCG.features.zero_cross_rate(startings, endings, filtered)
        time_3 = time.perf_counter()
        rampt = pyPCG.features.ramp_time(startings, endings, envelope)
        time_4 = time.perf_counter()
        peak_s = peak_spread_optimized(startings, endings, envelope.data,
                                       envelope.fs)
        time_5 = time.perf_counter()
        peakw = pyPCG.features.peak_width(startings, endings, envelope)
        time_6 = time.perf_counter()
        peak_c = pyPCG.features.peak_centroid(startings, endings, envelope)
        time_7 = time.perf_counter()

        print(f"Time delta: {time_2 - time_1}")
        print(f"Zero cross rate: {time_3 - time_2}")
        print(f"Ramp time: {time_4 - time_3}")
        print(f"Peak spread: {time_5 - time_4}")
        print(f"Peak width: {time_6 - time_5}")
        print(f"Peak centroid: {time_7 - time_6}")

        # frequency domain features
        time_1 = time.perf_counter()
        max_f = pyPCG.features.max_freq(startings, endings, filtered)
        time_2 = time.perf_counter()
        spec_sp = spectral_spread_optimized(startings, endings, filtered.data)
        time_3 = time.perf_counter()
        spec_w = pyPCG.features.spectral_width(startings, endings, filtered)
        time_4 = time.perf_counter()
        spec_c = pyPCG.features.spectral_centroid(startings, endings, filtered)
        time_5 = time.perf_counter()

        print(f"Max freq: {time_2 - time_1}")
        print(f"Spectral spread: {time_3 - time_2}")
        print(f"Spectral width: {time_4 - time_3}")
        print(f"Spectral centroid: {time_5 - time_4}")

        # wavelet based features
        # cwt_max=pyPCG.features.max_cwt(startings,endings,filtered)
        # peakdist_cwt=pyPCG.features.cwt_peakdist(startings,endings,filtered)
        time_1 = time.perf_counter()
        intens = pyPCG.features.dwt_intensity(startings, endings, filtered)
        time_2 = time.perf_counter()
        entropy = pyPCG.features.dwt_entropy(startings, endings, filtered)
        time_3 = time.perf_counter()

        print(f"DWT intensity: {time_2 - time_1}")
        print(f"DWT entropy: {time_3 - time_2}")

        # complexity based features
        time_1 = time.perf_counter()
        katz_frac = pyPCG.features.katz_fd(startings, endings, filtered)
        time_2 = time.perf_counter()
        # lyap=pyPCG.features.lyapunov(startings,endings,filtered)

        print(f"Katz fd: {time_2 - time_1}")

        if categorize:
            term_cat = categorize_pregnancy_term(pregnancy_term)
            BMI_cat=categorize_bmi(BMI)
            age_cat=categorize_age(age)
        else:
            term_cat = pregnancy_term
            BMI_cat = BMI
            age_cat = age

        # stat_list.append([term_cat,BMI_cat,age_cat,clin_hist, maxim, minim, mean, std, median])
        features_list.append(
            [term_cat, BMI_cat, age_cat, clin_hist, baby_gender, t_delt.mean(),
             zero_cr.mean(), rampt.mean(), peak_s.mean(), peakw.mean(),
             peak_c[0].mean(),
             max_f[0].mean(), spec_sp.mean(), spec_w.mean(), spec_c[0].mean(),
             intens.mean(), entropy.mean(), katz_frac.mean(), maxim, minim,
             mean, std, median])
        print(sound_with_data[i][-1])

    features_df = pd.DataFrame(features_list,
                               columns=["Pregnancy term category",
                                        "BMI category", "Age category",
                                        "Clinical history", "Baby gender",
                                        "Time delta", "Zero cross rate",
                                        "Ramp time",
                                        "Peak spread", "Peak width",
                                        "Peak centroid",
                                        "Max frequency", "Spectral spread",
                                        "Spectral width", "Spectral centroid",
                                        "DWT intensity", "DWT entropy",
                                        "Katz fraction dimension", "Minimum",
                                        "Maximum", "Mean", "STD", "Median"])

    if save:
        features_df.to_csv("processed.csv")


def read_data(filename, database):
    '''
    Reads in the data, and cleans up the inconsistencies that are in the original database.
    :param filename: The file the processed data from process are saved in.
    :param database: the fPCG database this comes from
    :return: cleaned up dataframe
    '''
    features_df = pd.read_csv(filename)

    #cleaning up the annotation of the data
    if database=="Shiraz":#so far I only processed Shiraz, If other databases also need cleaning,
        # I can write a code for that later
        features_df["Clinical history"] = features_df[
            "Clinical history"].str.replace(
            "Preterm rupture of membeans \(PROM\)", "PROM", regex=True
        )#Yes, it was written with a typo
        features_df["Clinical history"] = features_df[
            "Clinical history"].replace(
            [
                "Twins-admitted because one of the fetus grows slower",
                "Admitted in hospital because one of the twins grows slower",
                "Admitted in hospital due to slow growth of right fetus"
            ],
            "low fetal growth"
        )
        features_df["Clinical history"] = features_df[
            "Clinical history"].str.replace("Twins \(boys\)", "PROM",
                                            regex=True)
        #Only twin boys where they wrote "Twins \(boys\)" in the clinical history,
        #and it happened to have PROM, in general this would be stupid, but this is needed here

        #one hot encoding
        clinical_history_dummies = pd.get_dummies(
            features_df["Clinical history"], prefix="clinical_history"
        )
        features_df = pd.concat([features_df, clinical_history_dummies], axis=1)

        #dropping redundancies
        features_df = features_df.drop("Clinical history", axis=1)
        features_df = features_df.drop(
            "clinical_history_Recorded from the same subject as F93057", axis=1)
        features_df = features_df.drop(
            "clinical_history_Recorded from the same subject as F93065", axis=1)
        features_df = features_df.drop(
            "clinical_history_Recorded from the same subject as F93066", axis=1)
        # features_df = features_df.drop("clinical_history_Twins \(boys\)")
        features_df = features_df.drop("Unnamed: 0", axis=1)

        # Create a dictionary mapping old column names to new names
        new_column_names = {
            col: col.replace("clinical_history_", "")
            for col in features_df.columns
            if col.startswith("clinical_history_")
        }

        features_df = features_df.rename(columns=new_column_names)

        #cases where they wrote two diseases
        features_df["Decrease of amniotic fluid"] = (
                features_df["Decrease of amniotic fluid"]
                | features_df[
                    "Admitted in hospital due to decrease of amniotic fluid and low fetal growth"]
        ).astype(int)
        features_df["low fetal growth"] = (
                features_df["low fetal growth"]
                | features_df[
                    "Admitted in hospital due to decrease of amniotic fluid and low fetal growth"]
        ).astype(int)
        features_df = features_df.drop(
            "Admitted in hospital due to decrease of amniotic fluid and low fetal growth",
            axis=1
        )

        features_df["Epilepsy"] = (
            features_df["History of epilepsy and hypothyroidism"]
        ).astype(int)

        features_df["Hypothyroidism"] = (
                features_df["Hypothyroidism"]
                | features_df["History of epilepsy and hypothyroidism"]
        ).astype(int)
        features_df = features_df.drop(
            "History of epilepsy and hypothyroidism", axis=1
        )

        #pd.set_option("display.max_columns", None)
        #features_df

    return features_df

