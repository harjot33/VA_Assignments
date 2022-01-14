from imports import *
from assignments.assignment1.a_load_file import *
from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.d_data_encoding import *
from assignments.assignment1.e_experimentation import *

"""
The below method should:
?? subtask1  Handle any dataset (if you think worthwhile, you should do some pre-processing)
?? subtask2  Generate a (classification, regression or clustering) model based on the label_column 
             and return the one with best score/accuracy

The label_column can be categorical, numerical or None
-If categorical, run through ML classifiers in "a_classification" file and return the one with highest accuracy: 
    DecisionTree, RandomForestClassifier, KNeighborsClassifier or NaiveBayes
-If numerical, run through these ML regressors in "b_regression" file and return the one with least MSE error: 
    svm_regressor_1(), random_forest_regressor_1()
-If None, run through simple_k_means() and custom_clustering() and return the one with highest silhouette score.
(https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
"""


def generate_model(df: pd.DataFrame, label_column: Optional[pd.Series] = None) -> Dict:

    # model_name is the type of task that you are performing.
    # Use sensible names for model_name so that we can understand which ML models if executed for given df.
    # ex: Classification_random_forest or Regression_SVM.
    # model is trained model from ML process
    # final_score will be Accuracy in case of Classification, MSE in case of Regression and silhouette score in case of clustering.
    # your code here.

    # Here in this method, all the functionalities have been combined such that the model is selected based on the label column and the subsquent analysis is performed
    # accordingly
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    #Preprocessing
    for numeric_column in numeric_columns:
        df = fix_nans(df, numeric_column)
        df = fix_outliers(df, numeric_column)
        df[numeric_column] = normalize_column(df[numeric_column])

    for categorical_column in categorical_columns:
        df = fix_nans(df, categorical_column)
        df = fix_outliers(df, categorical_column)

    eps = [0.1, 0.4, 0.6, 0.7, 0.8]
    mins = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    if label_column is None:
        print("Clustering")
        for cc in categorical_columns:
            df = df.drop([cc],axis=1)
        kmeans_model = simple_k_means(df,3,"euclidean")
        dbscan_model = custom_clustering(df,0.4,2)


        if kmeans_model['score'] > dbscan_model['score']:
            model_name = "Kmeans Model"
            model = kmeans_model['model']
            final_score = kmeans_model['score']
            return dict(model_name=model_name, model=model, final_score=final_score)
        else:
            model_name = "DBSCAN Model"
            model = dbscan_model['model']
            final_score = dbscan_model['score']
            return dict(model_name=model_name, model=model, final_score=final_score)

    elif label_column.name in numeric_columns:
        print("ML Regression")
        for cc in categorical_columns:
            df = df.drop([cc],axis=1)
        ok = list(df.columns)
        if label_column.name in ok:
            Y = df[label_column.name]
            df = df.drop([label_column.name], axis=1)


        SVMReg = svm_regressor_1(df, Y)
        RFReg = random_forest_regressor_1(df, Y)

        if SVMReg['mse'] < RFReg['mse']:
            return dict(model_name="SVM_REG", model=SVMReg['model'], final_score=SVMReg['mse'])
        else:
            return dict(model_name="RF_REG", model=RFReg['model'], final_score=RFReg['mse'])

    elif label_column.name in categorical_columns:
        ok = list(df.columns)
        if label_column.name in ok:
            Y = df[label_column.name]
            df = df.drop([label_column.name], axis=1)
        score = 0;
        DCT = decision_tree_classifier(df, Y)
        RFC = random_forest_classifier(df, Y)
        KNNC = knn_classifier(df, Y)
        NBC = naive_bayes_classifier(df, Y)

        if DCT['accuracy'] >= RFC['accuracy'] and DCT['accuracy'] >= KNNC['accuracy'] and DCT['accuracy'] >= NBC['accuracy']:
            return dict(model_name="DECISIONTREE", model=DCT['model'], final_score=DCT['accuracy'])
        elif RFC['accuracy'] >= DCT['accuracy'] and RFC['accuracy'] >= KNNC['accuracy'] and RFC['accuracy'] >= NBC['accuracy']:
            return dict(model_name="RANDOMFOREST", model=RFC['model'], final_score=RFC['accuracy'])
        elif KNNC['accuracy'] >= RFC['accuracy'] and KNNC['accuracy'] >= DCT['accuracy'] and KNNC['accuracy'] >= NBC['accuracy']:
            return dict(model_name="KNN", model=KNNC['model'], final_score=KNNC['accuracy'])
        elif NBC['accuracy'] >= RFC['accuracy'] and NBC['accuracy'] >= KNNC['accuracy'] and DCT['accuracy'] >= DCT['accuracy']:
            return dict(model_name="NAIVE BAYES", model=NBC['model'], final_score=NBC['accuracy'])


    return dict(model_name=model_name, model=model, final_score=final_score)


def run_custom():
    start = time.time()
    print("Custom modeling in progress...")
    df = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'iris.csv'))
    result = generate_model(df,df.loc[:,"species"])
    print(f"result:\n{result}\n")

    end = time.time()
    run_time = round(end - start)
    print("Custom modeling ended...")
    print(f"{30 * '-'}\nCustom run_time:{run_time}s\n{30 * '-'}\n")


if __name__ == "__main__":
    run_custom()
