import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def piechart(dataframe: pd.DataFrame,
             field: str,
             cats: int = 5,
             unique_identifier: str = 'VIN') -> plt.pie:
    """
    Generate a pie chart and summary dataset.

    A quick way to analyze a field with multiple categorical features. This function ranks various categories
    into user-defined(var: cats) top categories by count and assigns rest categories into "others".
    Render the data in the Series as a matplotlib plot of the pie kind.

    :param dataframe: A pandas dataframe being worked with.
    :param field: The field needed to be analyzed.
    :param cats: Number of top categories needs to be classified
    :param unique_identifier: One unique identifier (field in dataset) which is never missing in the entire dataframe you are working with; especially where the analyzed field is not null.
    :return: A dataframe with top categories along with concentration % and a pie chart
    """
    pvt = dataframe.pivot_table(index=field, values=unique_identifier, aggfunc=np.count_nonzero).sort_values\
        (by=unique_identifier, ascending=False).reset_index().rename(columns={unique_identifier: 'Count'})
    pvt['Concentration'] = pvt.apply(lambda x: x.Count / pvt.Count.sum(), axis=1)
    top_n_cats = pvt[:cats]
    top_n_cats = top_n_cats.append(pd.DataFrame(data=[['Others', pvt.Count.sum() - top_n_cats.Count.sum(), 1.0 - top_n_cats.Concentration.sum()]]
                                                , columns=top_n_cats.columns))
    top_n_cats = top_n_cats.set_index(field)
    top_n_cats.plot.pie(y='Concentration', figsize=(10, 10), autopct='%.2f%%')
    plt.title(field.title() + ': Concentration', color='blue', size=20)
    plt.legend(loc='best')
    plt.show()
    return top_n_cats.style.format({'Concentration': '{:.2%}', 'Count': '{:,}'})

def barchart_continuous(dataframe: pd.DataFrame,
             field: str,
             binary_var: str,
             n: int = 8,
             min: float = None,
             max: float = None,
             label_true: str = None,
             label_false: str = None) -> plt.bar:
    """
    Generate a bar chart showing distribution of a binary variable.

    A quick way to analyze a continuous field's distribution across a binary variable. This function cuts the field into
     distinct buckets and renders the pair-wise data distribution as a matplotlib plot of the bar kind.
     Function allows zooming in to a specific data range using min, max function.(allows outlier ignorance)

    :param dataframe: A pandas dataframe being worked with.
    :param field: The field needed to be analyzed.
    :param binary_var: The binary variable across which the distribution is analyzed.
    :param n: number of buckets requested.
    :param min: Minimum of the data-range for zoom-in functionality.
    :param max: Maximum of the data-range for zoom-in functionality.
    :param label_true: Label for True binary class
    :param label_false: Label for False binary class
    :return: A grouped matplotlib barchart.
    """

    dataset_true = dataframe.loc[dataframe[binary_var] == 1]
    dataset_false = dataframe.loc[dataframe[binary_var] == 0]

    min = dataframe[field].min() if min is None else min
    max = dataframe[field].max() if max is None else max
    label_true = 'True' if label_true is None else label_true
    label_false = 'False' if label_false is None else label_false

    custom_bucket_array = np.linspace(min, max, n)
    dataset_true.loc[:, field] = pd.cut(dataset_true[field], custom_bucket_array)
    dataset_false.loc[:, field] = pd.cut(dataset_false[field], custom_bucket_array)

    categories = dataset_true[field].cat.categories
    ind = np.array([x for x, _ in enumerate(categories)])
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    plt.bar(ind, dataset_true.groupby(field).size() / dataset_true[field].count(), width, label=label_true)
    plt.bar(ind + width, dataset_false.groupby(field).size() / dataset_false[field].count(), width, label=label_false)

    plt.xticks(ind + width / 2, categories)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.legend(loc='best')
    plt.xticks(rotation=90)
    plt.title(field.title(), color='blue')
    plt.show()

def barchart_categorical(dataframe: pd.DataFrame,
                 field: str,
                 binary_variable: str) -> sns.barplot:
    """
    Generate a bar chart showing frequency and percentage of distribution of a binary variable.

    A quick way to analyze a continuous field's distribution across a binary variable. This function cuts the field into
     distinct buckets and renders the pair-wise data distribution as a matplotlib plot of the bar kind.
     Function allows zooming in to a specific data range using min, max function.(allows outlier ignorance)

    :param dataframe: A pandas dataframe being worked with.
    :param field: The field needed to be analyzed.
    :param binary_variable: The binary variable across which the distribution is analyzed.
    :return: A twin axis matplotlib (seaborn) barchart.
    """
    categories = dataframe.groupby([field])[binary_variable].agg(['count', 'sum', 'mean']).reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))

    sns.lineplot(x=categories.index, y=categories['mean'], ax=ax)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax2 = ax.twinx()
    sns.barplot(x=field, y="count", data=categories, ax=ax2, alpha=0.5)
    plt.show()