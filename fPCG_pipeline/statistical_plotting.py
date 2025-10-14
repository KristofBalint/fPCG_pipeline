import matplotlib.pyplot as plt
import seaborn as sns
from .preprocessing import *
from scipy import stats
import scikit_posthocs as sp

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