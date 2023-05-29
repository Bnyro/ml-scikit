import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn import linear_model
from sklearn import tree


def test_salary():
    df = pandas.read_csv("salary.csv", sep=',', on_bad_lines='skip')

    X = df.drop(columns=['Salary'])
    y = df['Salary']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    predictions = regr.predict(X_test)
    absolute_error = mean_absolute_error(y_test, predictions)
    print(f"Absolute error: {absolute_error}")


def test_happiness():
    df = pandas.read_csv('happiness.csv', sep=',', on_bad_lines='skip')

    X = df.drop(columns=['Country or region', 'Score', 'Overall rank'])
    y = df['Score']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True)

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    predictions = regr.predict(X_test)
    absolute_error = mean_absolute_error(y_test, predictions)
    print(f"Absolute error: {absolute_error}")


def test_music():
    df = pandas.read_csv("spotify.csv", sep=";", on_bad_lines='skip')

    X = df.drop(columns=['title', 'artist', 'top genre'])
    y = df['top genre']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True)

    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy score: {accuracy}")


test_music()
test_happiness()
test_salary()
