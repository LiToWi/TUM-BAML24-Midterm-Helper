# Other useful commands:
    # .shift() -> (ds["Min price"] - ds["Min price"].shift(1))
    # .diff() -> (ds["Min price"].diff())

# Useful python commands:
    # lambda x: x / 1.95583

# Merge options:
    # how='inner' -> only the common rows
    # how='outer' -> all rows
    # how='left' -> all rows from the left dataset
    # how='right' -> all rows from the right dataset

# Bayes: 
    # P(A|B) = P(B|A) * P(A) / P(B)

import enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import statsmodels.api as sm
import scipy.stats as dists

from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import norm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

class StatisticalTestHelper:
    dataset = None
    def __init__(self, path, delimiter=None, dataset=None):
        if dataset is not None:
            self.dataset = dataset
        elif delimiter is not None:
            self.dataset = pd.read_csv(path, delimiter=delimiter)
        else:
            self.dataset = pd.read_csv(path)

    def z_test(self, key, value, alternative='two-sided', alpha=0.05, print_result=False):
        sample_mean = self.dataset[key].mean()
        population_mean = value
        sample_std = self.dataset[key].std()
        sample_size = len(self.dataset[key])
        
        z_score = (sample_mean - population_mean) / (sample_std / np.sqrt(sample_size))
        
        if alternative == 'two-sided':
            p_value = 2 * (1 - norm.cdf(abs(z_score)))
            ci_low = sample_mean - norm.ppf(1 - alpha/2) * (sample_std / np.sqrt(sample_size))
            ci_high = sample_mean + norm.ppf(1 - alpha/2) * (sample_std / np.sqrt(sample_size))
        elif alternative == 'greater':
            p_value = 1 - norm.cdf(z_score)
            ci_low = sample_mean - norm.ppf(1 - alpha) * (sample_std / np.sqrt(sample_size))
            ci_high = np.inf
        elif alternative == 'less':
            p_value = norm.cdf(z_score)
            ci_low = -np.inf
            ci_high = sample_mean + norm.ppf(1 - alpha) * (sample_std / np.sqrt(sample_size))
        else:
            raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

        reject_null = p_value < alpha

        if print_result:
            print(f"Z-score: {z_score}")
            print(f"P-value: {p_value}")
            print(f"Confidence Interval: ({ci_low}, {ci_high})")
            print(f"Reject Null Hypothesis: {reject_null}")

        return z_score, p_value, (ci_low, ci_high), reject_null

    @staticmethod
    def t_test(a, b, test_kind="two-dependant-samples", alternative="greater", alpha=0.05, print_result=False):
        if test_kind == "two-dependant-samples" or test_kind == "paired":
            t_statistic , p_value = stats.ttest_rel (a , b, alternative=alternative)
            if print_result: print (" t - statistic = " , t_statistic)
            if print_result: print (" p - value = " , p_value)
            if print_result: print ("Rejection: " , p_value < alpha)

            return t_statistic, p_value, p_value < alpha
        
        elif test_kind == "two-independent-samples":
            t_statistic , p_value = stats.ttest_ind (a , b, alternative=alternative)
            if print_result: print (" t - statistic = " , t_statistic )
            if print_result: print (" p - value = " , p_value )
            if print_result: print ("Rejection: " , p_value < alpha)

            return t_statistic, p_value, p_value < alpha
        
        elif test_kind == "single-sample":
            t_statistic , p_value = stats.ttest_1samp (a , b, alternative=alternative)
            if print_result: print (" t - statistic = " , t_statistic )
            if print_result: print (" p - value = " , p_value )
            if print_result: print ("Rejection: " , p_value < alpha)

            return t_statistic, p_value, p_value < alpha
    
    @staticmethod
    def het_white_test(X, residuals, print_result=False):
        statistic, p_value, _, _ = het_white(residuals, X)
        if print_result: print(f"Value of the null-hypothesis that the residuals are homoscedastic: {statistic}")
        if print_result: print(f"p-value of the statistic: {p_value}")

    @staticmethod
    def get_vif(X, print_result=False):
        # VIF < 1: no correlation 
        # 1 < VIF < 5: moderate correlation, accepted
        # VIF > 5: high correlation, which may be problematic.
        # VIF > 10: very high correlation, indicating severe multicollinearity
        for index, variable_name in enumerate(X.columns):
            if variable_name == "const": 
                continue
            print(f"VIF for variable {variable_name} is {vif(X, index)}")

class ConfusionMatrixHelper:
    cm = None
    def __init__(self, y_true, y_pred):
        self.cm = confusion_matrix(y_true, y_pred)

        self.TP = self.cm[1, 1]
        self.TN = self.cm[0, 0]
        self.FP = self.cm[0, 1]
        self.FN = self.cm[1, 0]

        self.accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN)
        self.specificity = self.TN / (self.TN + self.FP)
        self.sensitivity = self.recall

        self.error_rate = 1 - self.accuracy
        self.false_positive_rate = 1 - self.specificity
        self.false_negative_rate = 1 - self.sensitivity
        self.true_positive_rate = self.sensitivity
        self.true_negative_rate = self.specificity

    def print_confusion_matrix(self):
        plt.matshow(self.cm, cmap='Blues')
        plt.colorbar(ticks=np.unique(self.cm))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


class PlotType(enum.Enum):
    LINE = 1
    SCATTER = 2
    BAR = 3
    HIST = 4

class DatasetHelper:
    dataset = pd.DataFrame()

    def __init__(self, dataset_src, delimiter=None, is_csv=True, sheet_name=None):
        if is_csv:
            self.dataset_src = dataset_src
            self.delimiter = delimiter
            self.reset_dataset()
        else:
            self.dataset = pd.read_excel(dataset_src, sheet_name=sheet_name)
            self.dataset.to_csv('convert.csv', index=False)
            self.dataset = pd.read_csv('convert.csv')


    ### Manage the dataset ###
    def get_dataset(self) -> pd.DataFrame:
        return self.dataset
    def set_dataset(self, dataset):
        self.dataset = dataset
    def reset_dataset(self) -> pd.DataFrame:
        self.dataset = pd.read_csv(self.dataset_src, delimiter=self.delimiter)
        return self.dataset
    
    ### Get information from the dataset ###
    def info(self, print_info=False):
        if print_info: print(self.dataset.info())
        return self.dataset.info()
    def get_column(self, key, start=0, end=-1, mask=None, print_column=False):
        if end == -1: end = len(self.dataset)
        if mask is None: mask = [True] * (end - start)
        res = self.dataset.loc[start:end, key][mask]
        if print_column: print(res)
        return res
    def get_column_names(self, print_columns=False):
        if print_columns: print(self.dataset.columns)
        return self.dataset.columns
    def get_row (self, key, print_row=False):
        if print_row: print(self.dataset.loc[key, :])
        return self.dataset.loc[key, :]
    def get_min_max(self, key, print_min_max=False):
        if print_min_max: print(f"min: {self.dataset[key].min()}, max: {self.dataset[key].max()}")
        return self.dataset[key].min(), self.dataset[key].max()
    def select_rows(self, key, condition, print_rows=False):
        if print_rows: print(self.dataset.loc[condition, key])
        return self.dataset.loc[condition, key]
    def get_argmax_row_of_key(self, key, print_row=False):
        if print_row: print(self.dataset.iloc[self.dataset[key].argmax()])
        return self.dataset.iloc[self.dataset[key].argmax()]
    def get_correlation_matrix(self, print_matrix=False):
        correlation_matrix = self.dataset.corr()
        if print_matrix: print(correlation_matrix)
        return correlation_matrix
    def get_diff(self, key, print_column=False):
        if print_column: print(self.dataset[key].diff())
        return self.dataset[key].diff()
    def get_percentage_diff(self, key, print_column=False):
        if print_column: print(self.dataset[key].diff() / self.dataset[key] * 100)
        return (self.dataset[key].diff() / self.dataset[key] * 100)
    def get_mean(self, key, mask, print_mean=False):
        if print_mean: print(self.dataset.loc[mask, key].mean())
        return self.dataset.loc[mask, key].mean()
    def get_variance(self, key, mask, print_variance=False):
        if print_variance: print(self.dataset.loc[mask, key].var())
        return self.dataset.loc[mask, key].var()
    def get_view(self, keys, print_column=False):
        if print_column: print(self.dataset[keys])
        return self.dataset[keys]
    def get_column_count_map(self, key, print_count=False):
        if print_count: print(self.dataset[key].value_counts())
        return self.dataset[key].value_counts()
    def get_column_count(self, key, print_count=False):
        if print_count: print(self.dataset[key].count())
        return self.dataset[key].count()
    def get_unique_values(self, key, print_values=False):
        if print_values: print(self.dataset[key].unique())
        return self.dataset[key].unique()
    
    @staticmethod
    def concat(datasets, print_column=False):
        dataset = pd.concat(datasets, axis=1)
        if print_column: print(dataset)
        return dataset
    
    # Aggregations
    def merge_datasets(self, other_dataset, on_key, left_on=None, right_on=None, how='inner', other_dataset_path=None):
        if other_dataset_path is not None: other_dataset = pd.read_csv(other_dataset)
        if left_on is not None and right_on is not None:
            self.dataset = pd.merge(self.dataset, other_dataset, left_on=left_on, right_on=right_on, how=how)
        else:
            self.dataset = pd.merge(self.dataset, other_dataset, on=on_key, how=how)
    def group_by(self, key, aggregation, reset_index, print_group=False):
        # Examplary groupby: ["GenreSong", "ArtistId"]
        # Examplary agg: MeanListen=('MinutesListened', 'mean'), MaxClick=('NumClicks', 'max')
        grouped = self.dataset.groupby(key).agg(aggregation) 
        if reset_index: grouped = grouped.reset_index() 
        if print_group: print(grouped)
        return grouped
    

    ### Add/change the dataset ###
    def add_column(self, key, value):
        self.dataset[key] = value
    def drop_column(self, key):
        self.dataset = self.dataset.drop(key, axis=1)
    def drop_row(self, index):
        self.dataset = self.dataset.drop(index)
    def apply_function_to_column(self, key, mask, function):
        self.dataset.loc[mask, key] = self.dataset.loc[mask, key].apply(function)
    def fill_nan(self, key, value):
        self.dataset[key].fillna(value, inplace=True)
    def map_values(self, key, dictionary):
        self.dataset[key] = self.dataset[key].map(dictionary)
    def drop_duplicates(self, subset=None):
        self.dataset = self.dataset.drop_duplicates(subset=subset)
    def drop_na(self):
        self.dataset = self.dataset.dropna()
    def rename_columns(self, key_mapping):
        self.dataset.rename(columns=key_mapping, inplace=True)
    def unify_labels(self, key, mapping):
        self.dataset[key] = self.dataset[key].map(mapping)
    def change_type(self, key, new_type):
        self.dataset[key] = self.dataset[key].astype(new_type)


    ### Visualize the dataset ###
    def print_head(self, *args, **kwargs):
        print(self.dataset.head(*args, **kwargs))
    

    def pretty_2d_plot(self, plotType, x_axis_name:str, y_axis_name:str, x_axis=None, y_axis=None, title="My Pretty Plot", legend={'show': False, 'label':None}):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

        if x_axis is None: x_axis = self.dataset[x_axis_name]
        if y_axis is None: y_axis = self.dataset[y_axis_name]
        
        plt.style.use('ggplot')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(x_axis_name, fontsize=12, fontweight='bold')
        plt.ylabel(y_axis_name, fontsize=12, fontweight='bold') 

        plt.grid(color='gray', linestyle='--', linewidth=0.5) 

        if plotType == PlotType.SCATTER:
            for i in range(len(y_axis)):
                plt.scatter(x_axis, y_axis[i], label=legend['labels'][i], color=colors[i])
        elif plotType == PlotType.LINE:
            for i in range(len(y_axis)):
                plt.plot(x_axis, y_axis[i], label=legend['labels'][i], color=colors[i])
        elif plotType == PlotType.BAR:
            bar_width = 0.35
            for i in range(len(y_axis)):
                plt.bar(x_axis + bar_width*i, y_axis[i], label=legend['labels'][i], color=colors[i], width=bar_width)

        if legend['show']: plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
        plt.show()


    ### Models ###
    def do_linear_regression(self, x_columns, y_column, added_X_params={}, print_result=False, plot_result=False):
        X = sm.add_constant(self.dataset[x_columns])
        for param in added_X_params:
            X[param] = added_X_params[param]
        y = self.dataset[y_column].values

        model = sm.OLS(y, X)
        results = model.fit()
        if print_result: print(results.summary())
        predicted_values = model.predict(results.params, X)
        residuals = self.dataset[y_column] - predicted_values

        # Visualization of the predicted variables vs. the true variables
        if plot_result:
            fig, axs = plt.subplots(1, 3, figsize=(20, 5))
            for ax, variable_name in zip(axs, x_columns):
                ax.scatter(self.dataset[variable_name], self.dataset[y_column], label="Ground truth")
                ax.scatter(self.dataset[variable_name], predicted_values, label="Model prediction")
                ax.legend()
                ax.set_xlabel(variable_name)
            plt.show()

        return model, predicted_values, residuals, X
    
    def do_logistic_regression(self, x_columns, y_column, print_result=False, plot_result=False):
        X = self.dataset[x_columns].copy()

        model = LogisticRegression()
        model.fit(X, self.dataset[y_column])

        beta_1, beta_2 = model.coef_[0]
        beta_0 = model.intercept_.item()
        if print_result: print(f"Betas: {beta_0}, {beta_1}, {beta_2}")

        self.dataset['prediction'] = model.predict(X)
        self.dataset['probability'] = model.predict_proba(X)[:,1]

        return model
    
    def random_forest(self, key, n_estimators=100, max_features=3, random_state=0, print_result=False):
        # 80/20 -> 80% training data for cross-validation, 20% test data as proxy after training
        # 4-fold cross-validation -> 4 different models with permutations of the training and validation data
        
        self.drop_na()
        # Identify non-numeric columns
        non_numeric_cols = self.dataset.select_dtypes(include=['object']).columns
        # Apply one-hot encoding to non-numeric columns
        self.dataset = pd.get_dummies(self.dataset, columns=non_numeric_cols, drop_first=True)
        
        X = self.dataset.drop(columns=[key])
        y = self.dataset[key]
        
        train_df, test_df = train_test_split(self.dataset, test_size=0.20, stratify=self.dataset[key], random_state=2023+2024)
        train_model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=random_state)

        cv_fits_accuracy = cross_val_score(train_model, X, y, cv=4, scoring='accuracy')
        cv_fits_precision = cross_val_score(train_model, X, y, cv=4, scoring='precision')
        cv_fits_recall = cross_val_score(train_model, X, y, cv=4, scoring='recall')

        print("\nCV-Accuracy:", np.mean(cv_fits_accuracy))
        print("CV-Precision:", np.mean(cv_fits_precision))
        print("CV-Recall:", np.mean(cv_fits_recall))

        # Train the final model
        train_model.fit(train_df.drop(columns=[key]), train_df[key])

        # Variable Importance Plot
        importance_values = train_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance_values})
        imp_plot = importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False)
        imp_plot.plot()
        plt.show()

        # Apply on test set
        test_predictions = train_model.predict(test_df.drop(columns=[key]))
        test_probabilities = train_model.predict_proba(test_df.drop(columns=[key]))

        test_predictions_df = pd.DataFrame({f'{key}': test_df[key], 
                                            f'Predicted_{key}': test_predictions,
                                            f'Probability_{key}=0': test_probabilities[:, 0],
                                            f'Probability_{key}=1': test_probabilities[:, 1]})

        # Confusion Matrix
        conf_matrix = ConfusionMatrixHelper(test_df[key], test_predictions)
        
        print("\nConfusion Matrix:")
        conf_matrix.print_confusion_matrix()

        # Precision, accuracy, recall
        print("\nTest-Precision:", conf_matrix.precision)
        print("Test-Accuracy:", conf_matrix.accuracy)
        print("Test-Recall:", conf_matrix.recall)


if __name__ == '__main__':
    ecom = DatasetHelper("e-commerce-dataset.xlsx", is_csv=False, sheet_name='E_Comm')
    ecom.change_type('Churn', 'category')

    ecom.random_forest('Churn', n_estimators=1000, max_features=3, random_state=0, print_result=True)


    