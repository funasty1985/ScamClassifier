import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def main():
    ## 1) Load the data into a pandas data frame

    filename = 'Youtube03-LMFAO.csv'
    data = pd.read_csv(filename)

    ## 2) initial data exploration
    print("#################################### Initial Data ###############################################")
    pd.set_option('display.max_columns', None)
    print("\nThe shape of the initial loaded data : ", data.shape)
    print("\nThe first five rows of the initial loaded data : ", data.head())
    # Show information about the data, such as column names and data types
    print("\ncolumns info of the initial loaded data : ")
    print(data.info())
    data = data[["CONTENT", "CLASS"]]

    ## 3) prepare the data for model building ## (done)
    data["CONTENT"] = data["CONTENT"].str.lower()
    stop_words = set(stopwords.words("english"))
    # remove stop words
    lancaster = LancasterStemmer()

    data["TOKENS"] = data["CONTENT"].apply(word_tokenize)
    data["TOKENS"] = data["TOKENS"].apply(lambda tokens: [token for token in tokens if token.lower() not in stop_words])
    data["TOKENS"] = data["TOKENS"].apply(lambda tokens: [lancaster.stem(token) for token in tokens])
    data["TOKENS_STR"] = data["TOKENS"].apply(lambda tokens: " ".join(tokens))

    # create the word_count vector
    vectorizer = CountVectorizer()
    init_features = vectorizer.fit_transform(data["TOKENS_STR"]).toarray()
    init_features_df = pd.DataFrame(init_features, columns=vectorizer.get_feature_names_out())

    ## 4) explore initial features
    print("\n\n#################################### Initial Feature ###############################################")
    print("\nThe shape of the initial feature : ", init_features_df.shape)
    print("\ncolumns info of the initial feature : ")
    print(init_features_df.info())
    print("\nunique values of the first column : ")
    print(init_features_df.iloc[:, 0].unique())
    print("\nunique vales of the first row:")
    print(list(set([ele for ele in init_features[0]])))

    print("Values bigger than zero:", [ele for ele in init_features[0] if ele > 0])
    print("Shape of FEATURES array:", init_features.shape)
    print("Size of vocabulary:", len(vectorizer.vocabulary_))
    print("Top 10 words in vocabulary:", list(vectorizer.vocabulary_.keys())[:10])

    number_of_unique_words = len(vectorizer.vocabulary_)

    print("\nNumber of unique words:", number_of_unique_words)

    non_zero_values = vectorizer.fit_transform(data["TOKENS_STR"]).data

    print("\nNon zero values:", non_zero_values)

    # print("\nFeature names:", vectorizer.get_feature_names())
    # print("\nFeature names:", vectorizer.get_feature_names_out())

    sparse_matrix_stats = pd.Series(non_zero_values).describe()

    print("\nSummary statistics of the sparse matrix:", sparse_matrix_stats)

    scam_words = ['huh', 'sexy', 'free gift', 'download', 'visit']
    print("Generated Scam Words:", scam_words)

    scam_data = []

    for i, row in data.iterrows():
        for word in scam_words:
            if word in row['CONTENT'].lower():
                scam_data.append(row)
                break

    # Print the number of rows of scam data found
    print(f"Found {len(scam_data)} rows of scam data")
    print(scam_data)

    ## 5) Downscale the transformed data ##  (final_features exploration incompleted) ##
    # used to find the importance of the words
    tfidf = TfidfTransformer()
    final_features = tfidf.fit_transform(init_features).toarray()
    scaled_features_df = pd.DataFrame(final_features)

    print("\n\n#################################### Scaled Feature ###############################################")
    print("\nThe shape of the scaled feature : ", scaled_features_df.shape)
    print("\ncolumns info of the scaled feature : ")
    print(scaled_features_df.info())
    print("\nunique values of the first column : ")
    print(scaled_features_df.iloc[:, 0].unique())
    print("\nunique vales of the first row:")
    print(list(set([ele for ele in final_features[0]])))

    ## 6) shuffle the dataset (done)  ##   (completed)
    # turn the final_features d-array to panada data_frame
    # concat scaled_features_df and data["CLASS"] to form a new data frame
    combined_df = pd.concat([scaled_features_df, data["CLASS"]], axis=1)
    # shuffle the data set
    combined_df = combined_df.sample(frac=1, random_state=1)

    ## 7) split the data withwout using test_train_ split  ##
    training_pd = combined_df.sample(frac=0.75, random_state=1)
    testing_pd = combined_df.drop(training_pd.index)

    X_train = training_pd.drop(columns=["CLASS"], axis=1)
    y_train = training_pd["CLASS"]
    X_test = testing_pd.drop(columns=["CLASS"], axis=1)
    y_test = testing_pd["CLASS"]

    ## 8) Fit the training data into a Naive Bayes classifier.  ##
    # build a model with the multinomial NB classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    ## 9) Cross validate
    print(
        "\n\n#################################### Cross Validation With Training Data ###############################################")
    num_folds = 5
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=num_folds)
    print(f"\n\n\nCross validation on training data")
    print(f"mean score: {scores.mean()}")
    print(f"minimum score: {scores.min()}")
    print(f"maximum scorce: {scores.max()}\n")

    ## 10) ## Test the model on the test data, print the confusion matrix and the accuracy of the model. ## (to do)
    print(
        "\n\n#################################### Testing With Testing Data ###############################################")
    y_pred = clf.predict(X_test)
    print("\nthe accuracy_score of test data : ", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title("Confusion matrix of the model on testing data")
    plt.show()

    ## 11) come up with 6 new comments (4 comments should be non spam and 2 comment spam)   ## (to do)
    print(
        "\n\n#################################### Testing With Custom Comments ###############################################")
    custom_data = pd.read_csv(r'custom_comment.csv', on_bad_lines='error')
    custom_data["CONTENT"] = custom_data["CONTENT"].str.lower()
    print(custom_data)
    stop_words = set(stopwords.words("english"))
    # remove stop words
    lancaster = LancasterStemmer()

    custom_data["TOKENS"] = custom_data["CONTENT"].apply(word_tokenize)
    custom_data["TOKENS"] = custom_data["TOKENS"].apply(
        lambda tokens: [token for token in tokens if token.lower() not in stop_words])
    custom_data["TOKENS"] = custom_data["TOKENS"].apply(lambda tokens: [lancaster.stem(token) for token in tokens])
    custom_data["TOKENS_STR"] = custom_data["TOKENS"].apply(lambda tokens: " ".join(tokens))

    # create the word_count vector
    # have to use old vectorizer, otherwise feature words count will be different
    # here we use transform but not transform_fit
    custom_data_init_features = vectorizer.transform(custom_data["TOKENS_STR"])

    # used to find the importance of the words
    custom_data_final_features = tfidf.transform(custom_data_init_features)
    y = custom_data["CLASS"]

    custom_y_pred = clf.predict(custom_data_final_features)
    cm = confusion_matrix(y, custom_y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title("Confusion matrix of the model on custom testing data")
    plt.show()

    ## 12) Present all the results and conclusions ##
    category_map = {
        "1": 'Spam',
        "0": 'Ham',
    }
    custom_y_pred_categories = [category_map[str(pred)] for pred in custom_y_pred]

    # Print the categories of the custom test data
    print("\n\nCustom Test Data Predictions:")
    for i in range(len(custom_data)):

        if custom_y_pred_categories[i] == 'Ham':
            print(f"{custom_data['CONTENT'][i]}: HAM")
        else:
            print(f"{custom_data['CONTENT'][i]}: SPAM")

    print("\n\nthe accuracy score of the custom test data : ", accuracy_score(y, custom_y_pred))

if __name__ == "__main__":
    main()
