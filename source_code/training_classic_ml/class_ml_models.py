# to process dataset
from sklearn.model_selection import train_test_split

# to train
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# to get performances
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class classification_models(object):

    def __init__(self, dataset, response):

        self.dataset = dataset
        self.response = response
        self.model = None

    def split_to_train(self, test_size=None):

        X_train, X_test, y_train, y_test = train_test_split(self.dataset, self.response, test_size=test_size, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def apply_model(self, X_train, y_train, option):
        
        if option == 1:
            self.model = RandomForestClassifier()
        elif option == 2:
            self.model = DecisionTreeClassifier()
        else:
            self.model = BaggingClassifier()
        
        self.model.fit(X_train, y_train)
    
    def get_performances(self, X_test, y_test):

        predictions = self.model.predict(X_test)

        accurcy_value = accuracy_score(y_test, predictions)
        precision_value = precision_score(y_test, predictions, average="weighted")
        recall_value = recall_score(y_test, predictions, average="weighted")
        f1_value = f1_score(y_test, predictions, average="weighted")

        performances = {"accuracy": accurcy_value, "precision": precision_value, "recall": recall_value, "f_score": f1_value}
        return performances
    



