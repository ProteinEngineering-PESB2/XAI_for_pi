# to process dataset
from sklearn.model_selection import train_test_split

# to train
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

# to get performances
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

class regression_models(object):

    def __init__(self, dataset, response):

        self.dataset = dataset
        self.response = response
        self.model = None

    def split_to_train(self, test_size=None):

        X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.response, test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def apply_model(self, X_train, y_train, option):
        
        if option == 1:
            self.model = RandomForestRegressor()
        elif option == 2:
            self.model = DecisionTreeRegressor()
        else:
            self.model = BaggingRegressor()
        
        self.model.fit(X_train, y_train)
    
    def get_performances(self, X_test, y_test):

        predictions = self.model.predict(X_test)

        r2_value = r2_score(y_test, predictions)
        mae_value = mean_absolute_error(y_test, predictions)
        mse_value = mean_squared_error(y_test, predictions)

        performances = {"r2_score": r2_value, "mean absolute error": mae_value, "mean squared error": mse_value}

        return performances