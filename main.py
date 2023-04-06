import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from nltk.stem.lancaster import LancasterStemmer


def main():
    ## 1) Load the data into a pandas data frame. ## (done)
    data = pd.read_csv(r'Youtube03-LMFAO.csv')

    ## 2) .....(To Do)

    ## 3) prepare the data for model building ## (done)
    pd.set_option('display.max_columns', None)
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

    ## 4) explore initial features (not completed)
    print("Values bigger than zero:", [ele for ele in init_features[0] if ele > 0])

    ## 5) Downscale the transformed data ##  (final_features exploration incompleted) ##
    # used to find the importance of the words
    tfidf = TfidfTransformer()
    final_features= tfidf.fit_transform(init_features)
    print("Values bigger tha zero after downscaling : ",[ele for ele in final_features.toarray()[0] if ele > 0])   ## compare to line 16

    ## 6) shuffle the dataset (done)  ##   (completed)
    # turn the final_features d-array to panada data_frame
    scaled_features_df = pd.DataFrame(final_features.toarray())
    # concat scaled_features_df and data["CLASS"] to form a new data frame
    combined_pd = pd.concat([scaled_features_df, data["CLASS"]], axis=1)
    # shuffle the data set
    combined_pd = combined_pd.sample(frac=1, random_state=1)

    ## 7) split the data withwout using test_train_ split  ##
    training_pd = combined_pd.sample(frac=0.75,random_state=1)
    testing_pd = combined_pd.drop(training_pd.index)

    X_train = training_pd.drop(columns=["CLASS"])
    y_train = training_pd["CLASS"]
    X_test = testing_pd.drop(columns=["CLASS"])
    y_test = testing_pd["CLASS"]

    ## 8) Fit the training data into a Naive Bayes classifier.  ##
    # build a model with the multinomial NB classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    ## 9) ## Cross validate ##         (to do)
    ## 10) ## Test the model on the test data, print the confusion matrix and the accuracy of the model. ## (to do)
    ## 11) come up with 6 new comments (4 comments should be non spam and 2 comment spam)   ## (to do)
    ## 12) Present all the results and conclusions ##

    ## useful for 10)
    # Compute prediction
    y_pred = clf.predict(X_test)
    print("the accuracy_score is : ", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot()

if __name__ == "__main__":
    main()

