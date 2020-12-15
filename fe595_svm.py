import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


def svm_output():
    stockdata = pd.read_csv("SPY_data.csv", index_col=0)

    # changes movement of stock prices to binary variables (1)->goes up, (0)->less than 0.5% movement (-1)->goes down
    def calc(x):
        if x > 0.005:
            return 1
        if x < -0.005:
            return -1
        else:
            return 0

    # 'move' uses calc function to determine of the stock moves up or down from the day before
    stockdata["Move"] = stockdata["SPY"].pct_change().apply(calc)
    # 'day shifted' moves the dates back one so that we predict the next days price
    stockdata["Day Shifted"] = stockdata["Move"].shift(-1)
    stockdata.dropna(inplace=True)

    # testing data is all the stock data in the year 2019
    testing = stockdata[stockdata.index >= "2019"]
    # training data every year but 2019, also drop SPY, Move and Day Shifted
    x_train = stockdata[stockdata.index < "2019"].drop(["SPY", "Move", "Day Shifted"], axis=1)
    y_train = stockdata[stockdata.index < "2019"]["Day Shifted"]
    x_testing = testing.drop(["SPY", "Move", "Day Shifted"], axis=1)
    y_testing = testing["Day Shifted"]

    svclassifier = SVC()
    svclassifier.fit(x_train, y_train)
    print("Score from SVM Model: ", svclassifier.score(x_testing, y_testing))
    y_pred = svclassifier.predict(x_testing)
    print(confusion_matrix(y_testing, y_pred))


def main():
    svm_output()


if __name__ == "__main__":
    main()
