import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textstat
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import japanize_matplotlib
from matplotlib_venn import venn2
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.utils.multiclass import unique_labels
sns.set()


def create_distplot(input_df):
    """[summary]
    数値データを統計量(mean, max, min, median, std)と共にプロットする関数
    また平均となる部分に対して縦線をプロット
    Args:
        input_df (pd.DataFrame): Target data
    """
    for column in input_df.select_dtypes(exclude=object):
        print(f"{column}: \n Mean:{input_df[column].mean() :0.3f}, Max:{np.max(input_df[column]) : 0.3f}, Min:{np.min(input_df[column]) : 0.3f}, Median:{np.median(input_df[column]) : 0.3f}, Std:{np.std(input_df[column]) : 0.2f}")
        sns.distplot(input_df[column])
        plt.axvline(input_df[column].mean(), color="red", label="mean")
        plt.grid()
        plt.legend()
        plt.show()
        print("***" * 40)


def create_value_count_plot(input_df, col_list, n=5):
    """
    指定したカラム(objectやcategory)データの数をプロットする関数

    args:
       input_df : pd.DataFrame
       col_list : list
       n : Argument to specify how many unique values should be plotted (default=5)
    """
    for col in col_list:
        plt.subplots(figsize=(8, 8))
        print(f"{col}: \n {input_df[col].value_counts()[:n]}")
        input_df[col].value_counts().plot.bar()
        plt.show()
        print("***" * 40)


def plot_intersection(left, right, column, set_labels, ax=None):
    left_set = set(left[column])
    right_set = set(right[column])
    venn2(subsets=(left_set, right_set), set_labels=set_labels, ax=ax)
    return ax


def plot_right_left_intersection(train_df, test_df, columns='__all__'):
    """2つのデータフレームのカラムの共通集合を可視化"""
    if columns == '__all__':
        columns = set(train_df.columns) & set(test_df.columns)

    columns = list(columns)
    nfigs = len(columns)
    ncols = 6
    nrows = - (- nfigs // ncols)
    fig, axes = plt.subplots(
        figsize=(3 * ncols, 3 * nrows), ncols=ncols, nrows=nrows)
    axes = np.ravel(axes)
    for c, ax in zip(columns, axes):
        plot_intersection(train_df, test_df, column=c,
                          set_labels=('Train', 'Test'), ax=ax)
        ax.set_title(c)
    return fig, ax


def visualize_binary_data(input_df, cols, target_cols, pos_var, neg_var):
    """
    二値分類用の割合データ

    cols:
      pos_var:問題設定となっている方のターゲット
      neg_var:pos_varとは違うターゲット
    """
    cross_table = pd.crosstab(
        input_df[cols], input_df[target_cols], margins=True)
    pos_name = cross_table[pos_var] / cross_table["All"]
    neg_name = cross_table[neg_var] / cross_table["All"]

    cross_table["pos_var"] = pos_name
    cross_table["neg_var"] = neg_name

    cross_table = cross_table.drop(index=["All"])
    display(cross_table)

    tmp = cross_table[["pos_var", "neg_var"]]
    tmp = tmp.sort_values(by="pos_var", ascending=False)
    tmp.plot.bar(stacked=True, title='pos_var vs neg bar by choice column.')
    plt.xlabel(cols)
    plt.ylabel("Percentage")
    plt.show()


def show_scatterplot(input_df, x, y, hue=None, reg=True, title=None, xlabel=None, ylabel=None):
    """[summary]
    plot scatter plot and correlation.
    Args:
        input_df (pd.DataFrame): target data
        x (string): target column name
        y (string): target column name
        hue (string, optional): sort specified data. Defaults to None.
        reg (bool, optional): plot regression. Defaults to True.
        title (string, optional): plot title. Defaults to None.
        xlabel (string, optional): plot label for target column named x. Defaults to None.
        ylabel (string, optional): plot label for target column named y. Defaults to None.
    """
    print(f"correlation: \n {input_df[[x, y]].corr()}")
    plt.figure(figsize=(8, 6))
    if hue is not None:
        input_df = input_df.sort_values(hue)
    if reg:
        sns.regplot(x=x, y=y, data=input_df, scatter=False, color="red")
    sns.scatterplot(data=input_df, x=x, y=y, hue=hue,
                    s=200, palette="Set1", alpha=0.5)
    if title is not None:
        plt.title(None)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()
    print("***" * 40)


def create_heatmap(input_df):
    """[summary]
    plot heatmap for integer and float column
    Args:
        input_df (pd.DataFrame): target data
    """
    plt.figure(figsize=(15, 15))
    corr = input_df.select_dtypes(exclude=object).corr()
    sns.heatmap(corr, annot=True, cmap='Blues', fmt=".3f")
    plt.show()


def create_static_feature_of_grk_feature(input_df, col, target, static='mean', n=None):
    """[summary]

    Args:
        input_df (pd.DataFrame): target data
        col (string): column
        target (string): target column
        static (str, optional): groupby statistics . Defaults to 'mean'.
        n (int, optional): Argument to specify how many plotted. Defaults to None.
    """
    grk_df = input_df.groupby(col).agg(
        {target: static}).sort_values(target, ascending=False)
    print("***" * 40)
    plt.figure(figsize=(10, 10))
    sns.barplot(data=pd.DataFrame(
        grk_df.values.T[:, :n], columns=grk_df.index.tolist()[:n]))
    plt.title(f"{static} feature of groupby {col} columns")
    plt.xlabel(f"unique feature for {col} colum")
    plt.ylabel(f"{target} {static}")
    plt.xticks(rotation=45)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Refer to: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, fontsize=25)
    plt.yticks(tick_marks, fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)
    plt.ylabel('True label', fontsize=25)
    plt.title(title, fontsize=30)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size="5%", pad=0.15)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=20)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           #            title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    fontsize=20,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


def get_basic_nlp_info(input_df, col, target):
    """[summary]
    create basic nlp feature like string length and sentence length
    moreover plot statistics for these nlp feature and heatmap
    Args:
        input_df (pd.DataFrame): target data
        col (string): string column
        target (string): target column
    """
    info_df = pd.DataFrame()
    info_df[target] = input_df[target]
    info_df["number_of_string"] = input_df[col].apply(lambda x: len(x))
    info_df["number_of_word_count"] = input_df[col].apply(
        lambda x: len(x.split()))
    info_df["number_of_unique_word_count"] = input_df[col].apply(
        lambda x: len(set(x.split())))
    info_df["number_of_sentence_count"] = input_df[col].apply(
        lambda x: textstat.sentence_count(x))
    info_df["number_of_syllable_count"] = input_df[col].apply(
        lambda x: textstat.syllable_count(x))
    info_df["avg_character_per_word"] = input_df[col].apply(
        lambda x: textstat.avg_character_per_word(x))
    info_df["avg_letter_per_word"] = input_df[col].apply(
        lambda x: textstat.avg_letter_per_word(x))
    info_df["avg_sentence_per_word"] = input_df[col].apply(
        lambda x: textstat.avg_sentence_per_word(x))
    info_df["avg_sentence_length"] = input_df[col].apply(
        lambda x: textstat.avg_sentence_length(x))
    display(info_df.describe())
    print("**" * 40 + "heatmap" + "**" * 40)
    create_heatmap(info_df)


def create_wordcloud(input_df, col, stopwords=None):
    """[summary]
    plot word frequency with WordCloud
    Args:
        input_df (pd.DataFrame): target data
        col (string): string column
        stopwords (list, optional): remove words . Defaults to None.
    """
    print(col)
    plt.subplots(figsize=(15, 15))
    wordcloud = WordCloud(stopwords=stopwords,
                          background_color="White", width=1800, height=1000)
    wordcloud.generate(" ".join(input_df[col]))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def create_top_ngram_word_plot(input_df, col, n=20, stop_words=None, ngram=4):
    """[summary]

    Args:
        input_df (pd.DataFrame): target data
        col (string): string column
        n (int, optional): Argument to specify how many data should be plotted. Defaults to 20.
        stop_words (list, optional): remove words. Defaults to None.
        ngram (int, n=1:unigram, n=2:bigram, n=3:trigram, n=4:all ngram(uni, bi, tri)): Specify which ngrams to plot. Defaults to 4.
    """
    def get_top_n_words(corpus, n=None, stop_words=None):
        vec = CountVectorizer(stop_words=stop_words).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def get_top_n_bigram(corpus, n=None, stop_words=None):
        vec = CountVectorizer(ngram_range=(
            2, 2), stop_words=stop_words).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def get_top_n_trigram(corpus, n=None, stop_words=None):
        vec = CountVectorizer(ngram_range=(
            3, 3), stop_words=stop_words).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx])
                      for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def plot_freq_word(df, n=None, name=None):
        plt.figure(figsize=(10, 10))
        ax = sns.barplot(x="freq", y="word", data=pd.DataFrame(
            df, columns=["word", "freq"]))
        plt.title(f"Top {n} {name}")
        plt.xticks(rotation=45)
        plt.xlabel("Frequency")
        plt.ylabel("")
        plt.show()

    if ngram == 1:
        unigram_df = get_top_n_words(input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(unigram_df, n=n, name="unigram")
    elif ngram == 2:
        bigram_df = get_top_n_bigram(input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(bigram_df, n=n, name="bigram")
    elif ngram == 3:
        trigram_df = get_top_n_trigram(
            input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(trigram_df, n=n, name="trigram")
    elif ngram == 4:
        print("***" * 40)
        unigram_df = get_top_n_words(input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(unigram_df, n=n, name="unigram")
        print("***" * 40)
        bigram_df = get_top_n_bigram(input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(bigram_df, n=n, name="bigram")
        print("***" * 40)
        trigram_df = get_top_n_trigram(
            input_df[col], n=n, stop_words=stop_words)
        plot_freq_word(trigram_df, n=n, name="trigram")


def visualize_importance(models, feat_train_df):
    """lightGBM の model 配列の feature importance を plot する
    CVごとのブレを boxen plot として表現します.

    args:
        models:
            List of lightGBM models
        feat_train_df:
            学習時に使った DataFrame
    """
    feature_importance_df = pd.DataFrame()
    for i, model in enumerate(models):
        _df = pd.DataFrame()
        _df['feature_importance'] = model.feature_importances_
        _df['column'] = feat_train_df.columns
        _df['fold'] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, _df],
                                          axis=0, ignore_index=True)

    order = feature_importance_df.groupby('column')\
        .sum()[['feature_importance']]\
        .sort_values('feature_importance', ascending=False).index[:50]

    fig, ax = plt.subplots(figsize=(8, max(6, len(order) * .25)))
    sns.boxenplot(data=feature_importance_df,
                  x='feature_importance',
                  y='column',
                  order=order,
                  ax=ax,
                  palette='viridis',
                  orient='h')
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Importance')
    ax.grid()
    return fig, ax
