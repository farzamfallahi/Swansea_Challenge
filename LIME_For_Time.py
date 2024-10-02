import numpy as np
from lime import lime_base
from lime.lime_tabular import LimeTabularExplainer

class LimeTimeSeriesExplainer(object):
    def __init__(self, training_data, feature_names, class_names=None, 
                 feature_selection='auto', num_features=None, num_samples=5000, 
                 random_state=None):
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.feature_selection = feature_selection
        self.num_features = num_features
        self.num_samples = num_samples
        self.random_state = random_state
        
        self.num_slices = training_data.shape[1]
        self.slice_names = [f"Slice {i+1}" for i in range(self.num_slices)]
        
    def explain_instance(self, instance, model, labels=(1,), num_features=None, 
                         num_samples=None, distance_metric='euclidean', model_regressor=None):
        num_features = self.num_features if num_features is None else num_features
        num_samples = self.num_samples if num_samples is None else num_samples
        
        data = self.training_data.reshape(self.training_data.shape[0], -1)
        
        explainer = LimeTabularExplainer(
            data, mode="regression", feature_names=self.slice_names,
            categorical_features=[], class_names=self.class_names,
            discretize_continuous=False, random_state=self.random_state
        )
        
        exp = explainer.explain_instance(
            instance.reshape(-1), model.predict, labels=labels, 
            num_features=num_features, num_samples=num_samples, 
            distance_metric=distance_metric, model_regressor=model_regressor
        )
        return exp

# Example usage
if __name__ == '__main__':
    # Assuming you have a trained time series model called 'model'
    # and the training data 'X_train' with shape (num_samples, num_timesteps, num_features)
    
    # Create an instance of LimeTimeSeriesExplainer
    explainer = LimeTimeSeriesExplainer(
        training_data=X_train.reshape(X_train.shape[0], -1),
        feature_names=[f"Feature {i+1}" for i in range(X_train.shape[2])],
        class_names=None,
        feature_selection='auto',
        num_features=10,
        num_samples=5000,
        random_state=42
    )
    
    # Select an instance to explain
    instance_index = 0
    instance = X_train[instance_index]
    
    # Generate the LIME explanation
    explanation = explainer.explain_instance(
        instance=instance,
        model=model,
        labels=(1,),
        num_features=10,
        num_samples=5000,
        distance_metric='euclidean',
        model_regressor=None
    )
    
    # Print the explanation
    print(explanation.as_list())