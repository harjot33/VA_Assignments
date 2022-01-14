from imports import *
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS, DBSCAN, AffinityPropagation, SpectralClustering
from sklearn import metrics
from assignments.assignment2.c_clustering import *
from assignments.assignment2.a_classification import *
from assignments.assignment2.b_regression import *
import pandas as pd
from assignments.assignment1.a_load_file import *
from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.d_data_encoding import *
from assignments.assignment1.e_experimentation import *
"""
Clustering is an unsupervised form of machine learning. It uses unlabeled data and returns the similarity/dissimilarity between rows of the data.
See https://scikit-learn.org/stable/modules/clustering.html for an overview of methods in sklearn.
"""


#######################################################
# Read the comments carefully in the following method(s)
# Assignment questions are marked as "??subtask"
#######################################################


def simple_k_means(x: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = KMeans(n_clusters=n_clusters)

    clusters = model.fit_transform(x)

    # There are many methods of deciding a score of a cluster model. Here is one example:
    score = metrics.silhouette_score(x, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


def iris_clusters() -> Dict:
    """
    Let's use the iris dataset and clusterise it:
    """
    iris = process_iris_dataset_again()
    iris.drop("large_sepal_length", axis=1, inplace=True)

    # Let's generate the clusters considering only the numeric columns first
    no_species_column = simple_k_means(iris.iloc[:, :4])

    ohe = generate_one_hot_encoder(iris['species'])
    df_ohe = replace_with_one_hot_encoder(iris, 'species', ohe, list(ohe.get_feature_names()))

    # Notice that here I have binary columns, but I am using euclidean distance to do the clustering AND score evaluation
    # This is pretty bad
    df_ohe = df_ohe.fillna(method='ffill')
    no_binary_distance_clusters = simple_k_means(df_ohe)


    # Finally, lets use just a label encoder for the species.
    # It is still bad to change the labels to numbers directly because the distances between them does not make sense
    print(iris.head(5))
    le = generate_label_encoder(iris['species'])
    df_le = replace_with_label_encoder(iris, 'species', le)
    labeled_encoded_clusters = simple_k_means(df_le)

    # See the result for yourself:
    r1 = round(no_species_column['score'], 2)
    r2 = round(no_binary_distance_clusters['score'], 2)
    r3 = round(labeled_encoded_clusters['score'], 2)
    print(f"Clustering Scores:\nno_species_column:{r1}, no_binary_distance_clusters:{r2}, labeled_encoded_clusters:{r3}")

    return max(r1, r2, r3)


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def custom_clustering(x: pd.DataFrame, eps: float, min_samples: float) -> Dict:
    """
    As you saw before, it is much harder to apply the right distance metrics. Take a look at:
    https://scikit-learn.org/stable/modules/clustering.html
    and check the metric used for each implementation. You will notice that suppositions were made,
    which makes harder to apply these clustering algorithms as-is due to the metrics used.
    Also go into each and notice that some of them there is a way to choose a distance/similarity/affinity metric.
    You don't need to check how each technique is implemented (code/math), but do use the information from the clustering
    lecture and check the parameters of the method (especially if there is any distance metric available among them).
    Chose one of them which is, in your opinion, generic enough, and justify your choice with a comment in the code (1 sentence).
    The return of this method should be the model, a score (e.g. silhouette
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) and the result of clustering the
    input dataset.
    """

    # I made use of the DBSCAN clustering algorithm as it is suitable for the task of clustering when the data has a large amount of
    # noise. In this algorithm, we can define the distance metric as a parameter.
    # I chose euclidean as the distance metric as I saw that it is the most consistent and best performing metric for this algorithm.
    # The algorithm is designed to be highly optimised for this metric and the computational costs are low as compared to the other metrics.
    dbmodel = DBSCAN(eps=eps, min_samples=min_samples,metric="euclidean")
    clusters = dbmodel.fit(x)
    labels = dbmodel.labels_
    score = metrics.silhouette_score(x, labels)


    return dict(model=dbmodel, score=score, clusters=clusters)


def cluster_iris_dataset_again() -> Dict:
    """
    Run the df returned form process_iris_dataset_again() method of A1 e_experimentation file through the custom_clustering and discuss (3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df1 = process_iris_dataset_again()

    # Firstly I removed the labels and the binary data from the dataframe as the custom clustering method uses an unsupervised learning algorithm so it doesn't require any labels.
    # Usage of labels in this algorithm would cause unnecessary noise in the data and cause incorrect clusters to be formed which ultimately would cause
    # lower silhoutte score (in the negatives would mean that the value has been assigned to a wrong cluster)
    # If the "Species" column was removed, there was only one cluster that was being formed.
    # Removed the binary data because it would hinder in choosing the correct epilson value.
    df1 = df1.drop(["large_sepal_length"], axis=1)
    eps = [0.4]
    mins = [1, 2, 3]
    for ep in eps:
        for n in mins:
            md = custom_clustering(df1, ep, n)
            model = md['model']
            score = md['score']
            clusters = md['clusters']
          #  print(model)
           # print(score)
            #print(clusters.labels_)

    # It was found that there were these clusters - [0,1,2]
    # The epsilon value and the min sample values was found to have not much effect on the score, however when these values were decreased in
    # decimals for eps, there was only single cluster or when the eps values were increased it also resulted in a single cluster only
    # There were no outliers, DBSCAN handled this data better than KMeans because of better handling of the outliers and overall noise impact.
    # The model made sense with a score of 74.2


    return dict(model=model, score=score, clusters=clusters)


def cluster_amazon_video_game() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset() methos of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """

    df2 = process_amazon_video_game_dataset()
    # Firstly remove the duplicates from the asin values as they are causing redundancy in the data and do not make any sense to keep it.
    # As our main goal is to cluster the data, getting rid of the asin values will help in accomplishing the task because we want to cluster the
    # similar users together, so keeping data which uniquely identifies them goes against the logic of clustering.
    # Dropped the time column as it has invalid type of data which doesn't suit the clustering format

    df2 = df2.drop('asin', axis=1)
    df2 = df2.drop('time',axis=1)
    df2 = df2.drop_duplicates()

    eps = [0.1, 0.4, 0.6, 0.7, 0.8]
    mins = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for ep in eps:
        for n in mins:
            md = custom_clustering(df2, ep, n)
            model = md['model']
            score = md['score']
            clusters = md['clusters']
            print(model)
            print(score)
            print(clusters.labels_)


    # The score reaches the highest point when the eps is 0.1 and the min samples are 9 -
    # Trying the same process again with normalization to see the difference.
    # Low number of samples fetch a low score.
    # When the number of samples increase, the score also rises.
    # The samples reach a saturation point, they are different for different epilsons.

    for c in list(df2.columns):
        df2[c] = normalize_column(df2[c])

    print("After normalization")
    for ep in eps:
        for n in mins:
            md = custom_clustering(df2, ep, n)
            model = md['model']
            score = md['score']
            clusters = md['clusters']
            print(model)
            print(score)
            print(clusters.labels_)

    # Due to making the data consistent, a high score of 68.7 was achieved at 0.1 EPS and 1 minimum samples.
    # This data is actually suffering from the curse of dimensionality, which causes the loss of information and then there is mixture in the seperate clusters
    # usage of the dimensionality reduction algorithms can help in solving the issue of mix clustering.
    # These observations are based on testing the model on different parameters i.e hyper parameter tuning to see how the model reacts to the change in the environment variables.

    return dict(model=None, score=None, clusters=None)


def cluster_amazon_video_game_again() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset_again() methos of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    # df3 = process_amazon_video_game_dataset_again()

    df2 = process_amazon_video_game_dataset()
    # Firstly remove the duplicates from the asin values as they are causing redundancy in the data and do not make any sense to keep it.
    # As our main goal is to cluster the data, getting rid of the asin values will help in accomplishing the task because we want to cluster the
    # similar users together, so keeping data which uniquely identifies them goes against the logic of clustering.
    # Dropped the time column as it has invalid type of data which doesn't suit the clustering format

    df2 = df2.drop('asin', axis=1)
    df2 = df2.drop('time', axis=1)
    df2 = df2.drop_duplicates()

    eps = [0.1, 0.4, 0.6, 0.7, 0.8]
    mins = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for ep in eps:
        for n in mins:
            md = custom_clustering(df2, ep, n)
            model = md['model']
            score = md['score']
            clusters = md['clusters']
            print(model)
            print(score)
            print(clusters.labels_)

    # The score reaches the highest point of 68. 7 when the eps is 0.1 and the min samples are 9 -
    # Trying the same process again with normalization to see the difference.
    # Low number of samples fetch a low score.
    # When the number of samples increase, the score also rises.
    # The samples reach a saturation point, they are different for different epilsons.
    # Model makes sense for this dataset
    # These observations are based on testing the model on different parameters i.e hyper parameter tuning to see how the model reacts to the change in the environment variables.


    for c in list(df2.columns):
        df2[c] = normalize_column(df2[c])

    print("After normalization")
    for ep in eps:
        for n in mins:
            md = custom_clustering(df2, ep, n)
            model = md['model']
            score = md['score']
            clusters = md['clusters']
            print(model)
            print(score)
            print(clusters.labels_)

    # Due to making the data consistent, a high score of 79.5 was achieved at 0.8 EPS and 2 minimum samples.
    # This highest scoring parameters have 36 clusters, with no outliers.
    # The clusters which have outliers have a low silhoutte score
    # With the increasing eps the score is also increasing with the samples being increased from  1.
    # This data is actually suffering from the curse of dimensionality, which causes the loss of information and then there is mixture in the seperate clusters
    # usage of the dimensionality reduction algorithms can help in solving the issue of mix clustering.
    # These observations are based on testing the model on different parameters i.e hyper parameter tuning to see how the model reacts to the change in the environment variables.
    return dict(model=None, score=None, clusters=None)


def cluster_life_expectancy() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset_again() methos of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    df4 = process_life_expectancy_dataset()
    # We drop the columns  - Country, Year - These do not make any sense to have, as we want to create clusters having similar information, so we
    # keeping the data which uniquely identifies rows goes against the principle of the clustering.
    # Time is not of relevance when it comes to clustering as the time has no impact on the rest of the data.
    # There are features which have more impact, we can learn about the importance of features by using GRIDSEARCHV like functionalities to see which
    # features have more impact.
    df4 = df4.drop(["country","year"], axis=1)
    eps = [0.1, 0.4, 0.6, 0.7, 0.8]
    mins = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for ep in eps:
        for n in mins:
            md = custom_clustering(df4, ep, n)
            model = md['model']
            score = md['score']
            clusters = md['clusters']
            print(model)
            print(score)
            print(clusters.labels_)
    return dict(model=None, score=None, clusters=None)


def run_clustering():
    start = time.time()
    print("Clustering in progress...")
    assert iris_clusters() is not None
    assert len(cluster_iris_dataset_again().keys()) == 3
    #assert len(cluster_amazon_video_game().keys()) == 3
    #assert len(cluster_amazon_video_game_again().keys()) == 3
    #assert len(cluster_life_expectancy().keys()) == 3

    end = time.time()
    run_time = round(end - start, 4)
    print("Clustering ended...")
    print(f"{30*'-'}\nClustering run_time:{run_time}s\n{30*'-'}\n")


if __name__ == "__main__":
    run_clustering()
