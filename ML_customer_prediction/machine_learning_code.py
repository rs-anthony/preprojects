import sklearn
import numpy as np
import pandas as pd
from numpy import nan
from matplotlib.pyplot import figure
from sklearn import metrics
from datetime import date
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
#from imblearn.over_sampling import SMOTE
import numpy
import pickle

plt.style.use("ggplot")

RSEED = 10

### Load the merged file into a dataframe
df1 = pd.read_csv("")

'''

Features used in the model

'''
##  In this section you can add or remove features that are to be used in the model
categorical_features = ['Client',
                        'newsletter','how_unknown','partner_sort_income', 'Gender', 
                        'partner_gender','partner_marital_status_differs', 'partner_marital_status',
                        'partner_previous_marriage', 'partner_education_level', 'partner_type_ondernemer',
                        'borrowed_money', 'partner_kvk_subscription', 'partner debts or liabilities',
                        'open_credits', 'overdraft/creditcard', 'debts or liabilities', 'Student loans', 
                        'type_ondernemer', 'kvk_subscription', 'sort_income', 'previous_marriage', 'marital_status',
                        'education_level', 'partner_open_credits', 'partner_overdraft/creditcard', 'partner_borrowed_money',
                        'previous_client', 'penalty_amount',
                        
                        'Is/Was the current/previous property located in The Netherlands?',
                        'hypotheek_woning_waar_je_zelf_woont_combined', 'nationality', 'advice_about', 
                        'voldoende_eigen_middelen', 'iwantmortgage_combined', 'on_funda',
                        'In which country is/was the current/previous property located?', 'url_nieuwbouw_filled', 
                        'hypotheekrente_situatie', 'alleen_kopen', 'Allowed to have overdraft / I have a credit card',
                        'One or more credits open', 'Are you already a unknown customer?', 
                        'nieuw_bestaande_bouw_combined', 'Ingevuld door HF?', 'applicable_combined',
                        'eerder_koopwoning_combined', 'Borrowed money from family / others', 'which_social_medium',
                        'hypotheek_alleen', 'No debts or liabilities', 'Mortgage_arranged_unknown_combined',
                        'hypotheek_verhogen_via_ons_geregeld']

numerical_features = ['client_age','partner_age']

'''

Apply modifications to application form data

'''
# Creation of 'Client' column which is the target of the models
df1["Client"] = np.where( (df1["Datum getekende OTD"].replace({ np.nan : 0}) == 0), 0, 1)

### Nationality split into groups holland, europe and other
df1["nationality"] = df1["nationality"].str.lower()
df1['In which country is/was the current/previous property located?'] = df1['In which country is/was the current/previous property located?'].str.lower()

# The replacements of holland and europe are specified here
replace_nat = {'netherlands' : 'NL', 'nederland' : 'NL', 'dutch' : 'NL', 'nl' : 'NL', 
                'holland' : 'NL', 'nederlands' : 'NL', 'hollands' : 'NL', 'austria':'EU', 'italy':'EU', 'belgium':'EU',              
              'latvia':'EU', 'bulgaria':'EU', 'lithuania':'EU', 'croatia':'EU', 'luxembourg':'EU', 'cyprus':'EU', 'malta':'EU',
              'czechia':'EU', 'denmark':'EU', 'poland':'EU', 'estonia':'EU', 'portugal':'EU', 'finland':'EU', 'romania':'EU',
              'france':'EU', 'slovakia':'EU', 'germany':'EU', 'slovenia':'EU', 'greece':'EU', 'spain':'EU', 'hungary':'EU', 
               'sweden':'EU', 'ireland':'EU', 'united kingdom':'EU', 'engeland':'EU'}

# Changes in the dataframe of the nationality replacements
df1 = df1.replace({"nationality": replace_nat})
df1["nationality"] = np.where( (df1["nationality"] != 'NL') & (df1["nationality"] != 'EU'), 'other', df1["nationality"])
df1 = df1.replace({'In which country is/was the current/previous property located?':replace_nat})
df1['In which country is/was the current/previous property located?'] = np.where( 
    (df1['In which country is/was the current/previous property located?'] != 'NL') & 
    (df1['In which country is/was the current/previous property located?'] != 'EU'),
    'other', df1['In which country is/was the current/previous property located?'])

# Nationality partner NL, EU, other
df1["partner_nationality"] = df1["partner_nationality"].str.lower()
df1 = df1.replace({"partner_nationality": replace_nat})
df1["partner_nationality"] = np.where( (df1["partner_nationality"] != 'NL') & (df1["partner_nationality"] != 'EU'), 'other', df1["partner_nationality"])


## Features are combined, null values get removed and other specific modifications are made

# Combine applicable_situation and applicable_situation_other
df1["applicable_combined"] = df1["applicable_situation"].replace(np.nan, '', regex=True) + df1["applicable_situation_other"].replace(np.nan, '', regex=True) 

# Combine applicable_situation and applicable_situation_other
df1["nieuw_bestaande_bouw_combined"] = df1["nieuw_bestaande_bouw"].replace(np.nan, '', regex=True) + df1["nieuw_bestaande_bouw.1"].replace(np.nan, '', regex=True) 

# nieuwbouw filled is filled if one of two nieuwbouw columns has input
df1["url_nieuwbouw"] = df1["url_nieuwbouw"].replace({ np.nan : 0})
df1["url_nieuwbouw.1"] = df1["url_nieuwbouw.1"].replace({ np.nan : 0})
df1["url_nieuwbouw_filled"] = np.where( (df1["url_nieuwbouw"] == 0) & (df1["url_nieuwbouw.1"] == 0), 0, 1)

# city_new_house modifications: lowercase. Added city_new_house_filled variable
df1["city_new_house"] = df1["city_new_house"].str.lower()
df1["city_new_house_filled"] = np.where( (df1["city_new_house"].replace({ np.nan : 0}) == 0), 0, 1)

# Merge all three "i want a mortgage" kolommen. Now stacks the answers if multiple answers are given,
# these will become seperate categories
df1["iwantmortgage_combined"] = df1["I want a mortgage on the new property to be used as a rental unit"].replace(np.nan, '', regex=True) + df1["I want a mortgage on a property that I already own which is used as a rental unit"].replace(np.nan, '', regex=True) + df1["I want a mortgage or credit line on the home that I live in"].replace(np.nan, '', regex=True) 

# eerder_koopwoning_combined: eerder_koopwoning_gehad_2_personen + eerder_koopwoning_gehad_1_persoon. 
df1["eerder_koopwoning_combined"] = df1["eerder_koopwoning_gehad_1_persoon"].replace(np.nan, '', regex=True) + df1["eerder_koopwoning_gehad_2_personen"].replace(np.nan, '', regex=True)

# hypotheek_woning_waar_je_zelf_woont_combined has 2 features with the same name, they get combined
df1["hypotheek_woning_waar_je_zelf_woont_combined"] = df1["hypotheek_woning_waar_je_zelf_woont"].replace(np.nan, '', regex=True) + df1["hypotheek_woning_waar_je_zelf_woont_2"].replace(np.nan, '', regex=True) 

# Combine mortgage arranged/not arranged by unknown into one column
df1["Mortgage_arranged_unknown_combined"] = df1["Mortgage arranged by unknown"].replace(np.nan, '', regex=True) + df1["Mortgage not arranged by unknown"].replace(np.nan, '', regex=True) 

# Some values were english others where dutch, streamlined to english
df1["Gender"] = df1["Gender"].replace("Vrouw", "Female", regex=True)
df1['Gender'].replace('Vrouw','Female', inplace=True)

# combining how_unknown column and streamlining 
df1['how_unknown'] = np.where(df1['how_unknown'] == 'Via sociale media', 
                      np.where(df1['which_social_medium'] != nan, df1['which_social_medium'], 'andere sociale media'),
                      df1['how_unknown'])
df1['how_unknown'].replace('Overig','andere sociale media', inplace=True)

# simplifying values
df1['partner_sort_income'].replace('Mijn partner heeft een vaste arbeidsovereenkomst','Vaste arbeidsovereenkomst', inplace=True)
df1['partner_sort_income'].replace('Mijn partner heeft een tijdelijke arbeidsovereenkomst, <i>zonder</i> intentieverklaring','Tijdelijke overeenkomst zonder intentieverklaring', inplace=True)
df1['partner_sort_income'].replace('Mijn partner heeft een tijdelijke arbeidsovereenkomst <i>met</i> intentieverklaring','Tijdelijke overeenkomst met intentieverklaring', inplace=True)
df1['partner_sort_income'].replace('Mijn partner heeft geen inkomen','Geen', inplace=True)

df1['sort_income'].replace('Ik heb een vaste arbeidsovereenkomst','Vaste arbeidsovereenkomst', inplace=True)
df1['sort_income'].replace('Ik heb een tijdelijke arbeidsovereenkomst, <i>zonder</i> intentieverklaring','Tijdelijke overeenkomst zonder intentieverklaring', inplace=True)
df1['sort_income'].replace('Ik heb een tijdelijke arbeidsovereenkomst <i>met</i> intentieverklaring','Tijdelijke overeenkomst met intentieverklaring', inplace=True)
df1['sort_income'].replace('Ik ben ondernemer','Ondernemer', inplace=True)
df1['sort_income'].replace('Ik heb inkomen uit pensioen','Inkomen uit pensioen', inplace=True)
df1['sort_income'].replace('Ik heb geen inkomen','Geen', inplace=True)

### Features that contain not filled in/filled in setup, changed to yes/no
yes_no_features = {'Borrowed money from family / others':'borrowed_money','One or more credits open':'open_credits',
                  'Allowed to have overdraft / I have a credit card':'overdraft/creditcard', 'Student loans':'Student loans',
                   'Student loans.1':'partner student loans', 'One or more credits open.1': 'partner_open_credits', 
                   'Allowed to have overdraft / has a credit card' : 'partner_overdraft/creditcard', 
                   'Borrowed money from family / others.1' : 'partner_borrowed_money'
                  }

def simply_yes_no(dfr,infeature,outfeature):
    dfr[infeature].replace(nan,'nan',inplace=True)
    df1[outfeature] = np.where(df1[infeature] == 'nan', 'no', 'yes')
    
for k,v in yes_no_features.items():
    simply_yes_no(df1,k,v)

# Reverse yes to no for specific features
no_means_yes = {'No debts or liabilities':'debts or liabilities', 'No debts or liabilities.1':'partner debts or liabilities'}

def switch_yes_no(dfr,infeature,outfeature):
    dfr[infeature].replace(nan,'nan',inplace=True)
    df1[outfeature] = np.where(df1[infeature] == 'nan', 'yes', 'no')
    
for k,v in no_means_yes.items():
    switch_yes_no(df1,k,v)
    
#replace nan in categorical variables and make all English
for feature in categorical_features:
    df1[feature].replace(nan,'No_input', inplace=True)
    df1[feature].replace('ja','yes',inplace=True)
    df1[feature].replace('Ja','yes',inplace=True)
    df1[feature].replace('nee','no',inplace=True)
    df1[feature].replace('Nee','no',inplace=True)
    
### Birthdates to ages
feature_in_out = {'birthdate' : 'client_age', 'partner_birthdate' : 'partner_age'}

def date_to_age(dfr,birthdate,age):
    now = pd.Timestamp('now')
    dfr[age] = np.where(dfr[birthdate] != nan, pd.to_datetime(dfr[birthdate], format='%d-%m-%Y'),np.datetime64('NaT'))
    dfr[age].replace(np.datetime64('NaT'),now,inplace=True)
    dfr[age] = (now - dfr[age]).astype('<m8[Y]')
    #print(pd.unique(dfr[age]))

for k,v in feature_in_out.items():
    date_to_age(df1,k,v)
    
### Outlier check of specific features
factor = 3
possible_outliers = ['client_age', 'partner_age']
df2 = df1
for feature in possible_outliers:
    upper_lim = df2[feature].mean() + df2[feature].std() * factor
    lower_lim = df2[feature].mean () - df2[feature].std () * factor
    df2 = df2[(df2[feature] < upper_lim) & (df2[feature] > lower_lim)]
print(len(df1),'=>',len(df2))

### Unique value check per feature
# Remove the commented code for a list all unique values with the amount of times that they occur.

## count: The maximum amount of unique values in a feature that is printed. Changing this to 0 means never printing 
## the actual unique values but only the amount of unique values in that feature
for column in df1:
    uniq_val = df1[column].unique()
    count = len(uniq_val)
    #if count < 15:
        #print("The unique values in {} are: {}".format(column, uniq_val))
    #else: 
        #print("{} has {} unique values".format(column, count))
        
        
'''
Generating data
'''
### Creation of dummies
important_features = categorical_features + numerical_features
df1_important = df1[important_features]

df1_dum = pd.get_dummies(df1_important, columns = categorical_features)
df1_dum.pop('Client_0')
df1_dum.pop('Client_1')
cols = df1_dum.columns.tolist()

### Print of all created dummies
for f in cols:
    print(f)
    
    
'''
Creation of training, validation and testing split in data
'''
# export column for training as reference
with open('columns.pickle', 'wb') as c:
    pickle.dump(cols, c)
    
### Creation of testing set
lb = LabelBinarizer()
df1_dum['Client'] = lb.fit_transform(df1['Client'].values)
labels = df1_dum['Client']

df1_dum.drop(['Client'], axis=1, inplace=True)

train, test_final, train_labels, test_labels_final = train_test_split(df1_dum, labels, 
                                                          stratify = labels,
                                                          test_size = 0.3, 
                                                          random_state = RSEED)
features = list(train.columns)

### Creation of training and validation sets
x_train, test, y_train, test_labels = train_test_split(train, train_labels,
                                                  test_size = .1,
                                                  random_state=RSEED)

### SMOTE for balancing the dataset
sm = 0 #SMOTE(random_state=RSEED, sampling_strategy = 1.0)
train, train_labels = sm.fit_sample(x_train, y_train)

# print(x_train_res['Client'].value_counts())
print(len(x_train))
unique, counts = numpy.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))
unique1, counts1 = numpy.unique(train_labels, return_counts=True)
print(dict(zip(unique1, counts1)))

'''
Model training and testing
'''

##### DECISION TREE #####
# Create decision tree classifier and fit the model
d_tree = DecisionTreeClassifier(max_depth=5, random_state=RSEED)
d_tree.fit(train, train_labels)

## Make predictions of the probability that a datapoint is assigned to be a client
train_probs = d_tree.predict_proba(train)[:, 1]
val_probs = d_tree.predict_proba(test)[:, 1]
probs = d_tree.predict_proba(test_final)[:, 1]

## Predict whether a datapoint is a client
train_predictions = d_tree.predict(train)
val_predictions = d_tree.predict(test)
predictions = d_tree.predict(test_final)

# Print result metrics: ROC AUC and the model accuracy
print(f'Train ROC AUC Score: {roc_auc_score(train_labels, train_probs)}')
print(f'Validation ROC AUC  Score: {roc_auc_score(test_labels, val_probs)}')
print(f'Baseline validation ROC AUC: {roc_auc_score(test_labels, [1 for _ in range(len(test_labels))])}')
print(f'Test ROC AUC  Score: {roc_auc_score(test_labels_final, probs)}')
print(f'Baseline test ROC AUC: {roc_auc_score(test_labels_final, [1 for _ in range(len(test_labels_final))])}')

print(f'\n Model Accuracy: {d_tree.score(train, train_labels)}')

## Function used to evaluate the model by comparing to baseline performance
def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels_final, [1 for _ in range(len(test_labels_final))])
    baseline['precision'] = precision_score(test_labels_final, [1 for _ in range(len(test_labels_final))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels_final, predictions)
    results['precision'] = precision_score(test_labels_final, predictions)
    results['roc'] = roc_auc_score(test_labels_final, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels_final, [1 for _ in range(len(test_labels_final))])
    model_fpr, model_tpr, _ = roc_curve(test_labels_final, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    
## Evaluation of the Decision Tree
evaluate_model(predictions, probs, train_predictions, train_probs)

print("Training accuracy is: ", metrics.accuracy_score(train_labels, d_tree.predict(train)))
print("Test accuracy is: ", metrics.accuracy_score(test_labels_final, d_tree.predict(test_final)))

## Top Feature importances in the decision tree
# Display the 10 features with the highest feature importance in the Decision tree
fi_model = pd.DataFrame({'feature': features,
                   'importance': d_tree.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi_model.head(10)

##### Random Forest #####

# compares different hyperparameter values of the model. 
# Returns an object where the optimal parameters are stored in its methods.
# via the refit_score argument the optimization focus can be set, for recall, precision or accuracy
# More parameters require exponentially more computing time

from sklearn.ensemble import RandomForestClassifier

df1_forest = RandomForestClassifier(random_state=RSEED, n_jobs=-1)

param_grid = {
        'min_samples_split': [2, 3, 4],
        'n_estimators' : [140, 150, 160],
        'max_depth': [30, 35, 40],
        'max_features': [150, 170, 190],
        'class_weight': [None, 'balanced', {0:1, 1:5}]
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}

def grid_search_wrapper(refit_score='recall_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(df1_forest, param_grid, scoring=scorers, refit=refit_score,
                           cv=skf, return_train_score=True, n_jobs=-1)
    grid_search.fit(train.values, train_labels.values)

    # make the predictions
    target_pred = grid_search.predict(test_final.values)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(test_labels_final, target_pred),
                 columns=['pred_Not_Client', 'pred_Client'], index=['Not_Client', 'Client']))
    return grid_search


grid_search_df1_forest = grid_search_wrapper(refit_score='recall_score')

# Exports the trained object (model) for reference later (by the predictor)
with open('grid_search_object.pickle', 'wb') as f:
    pickle.dump(grid_search_df1_forest, f)
    
    # Displays the chosen parameters, as well as the scores.
results = pd.DataFrame(grid_search_df1_forest.cv_results_)
results = results.sort_values(by='mean_test_recall_score', ascending=False)
results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_max_depth',
         'param_max_features', 'param_min_samples_split', 'param_n_estimators']].round(3).head()

target_scores = grid_search_df1_forest.predict_proba(test_final)[:, 1]

p, r, thresholds = precision_recall_curve(test_labels_final, target_scores)


def adjusted_classes(target_scores, t):
    return [1 if y >= t else 0 for y in target_scores]

def precision_recall_threshold(p, r, thresholds, t=0.5):
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(target_scores, t)
    print(pd.DataFrame(confusion_matrix(test_labels_final, y_pred_adj),
                       columns=['pred_Not_Client', 'pred_Client'], 
                       index=['Not_Client', 'Client']))
    
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.0, 1.01]);
    plt.xlim([0.0, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)
    
precision_recall_threshold(p, r, thresholds, 0.40)

#### Plots the progression of the precision and recall score as a function of the decision threshold.

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    
#     Sets the figure size and title of the plot
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    
#     Plots the progression of the precision score
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    
#     Plots the progression of the recall score
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    
#     Labels the axes and assigns a place to the legend
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
    
plot_precision_recall_vs_threshold(p, r, thresholds)

# fits the classifier on the training data and calculates the average number of nodes
# and  average maximum depth of the forest

# Fit classifier on training data
df1_forest.fit(train, train_labels)
n_nodes = []
max_depths = []

# for each tree, collect its number of nodes and maximum depth and save them in a list
for ind_tree in df1_forest.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
# print the average number of nodes and maximum depth
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

##### Prints the recall score of the Validation test set and the Final test set.

print ('Validation Results')
print (recall_score(test_labels, df1_forest.predict(test)))
print ('\nTest Results')
print (recall_score(test_labels_final, df1_forest.predict(test_final)))

##### Prints the scores and plots of the dataset, using multiple scoring methods.

# Prints the accuracy of the model on the training and test set
print("Training accuracy is: ", metrics.accuracy_score(train_labels, df1_forest.predict(train)))
print("Test accuracy is: ", metrics.accuracy_score(test_labels_final, df1_forest.predict(test_final)))

# Collects the predictions and probability of correctness from the Random Forest on the training data and saves them
train_rf_predictions = df1_forest.predict(train)
train_rf_probs = df1_forest.predict_proba(train)[:, 1]

# Collects the predictions and probability of correctness from the Random Forest on the test data and saves them
rf_predictions = df1_forest.predict(test_final)
rf_probs = df1_forest.predict_proba(test_final)[:, 1]

# Evaluates the model using the function defined earlier
evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)

# Collects the feature importance of each feature in the dataset, sorts them from most important.
# to least important and prints it

fi_model = pd.DataFrame({'feature': features,
                   'importance': df1_forest.feature_importances_}).\
                    sort_values('importance', ascending = False)
fi_model.head(-1)
    
###### Defines the Random Forest that will be used and seperates the target class by it's values.

# Initiates the random forest and sets its parameters
test_forest = RandomForestClassifier(random_state=RSEED, n_jobs=-1, min_samples_split=3, 
    n_estimators=150, max_depth=35, max_features=170, class_weight='balanced')

# Counts the number of occurences of each class
count_class_0, count_class_1 = labels.value_counts()

# Sets up the dataset for the undersampling
df_test = df1_dum
df_test['Client'] = lb.fit_transform(df1['Client'].values)

# Divides the dataset by the values of the target class
df_class_0 = df_test[df_test['Client'] == 0]
df_class_1 = df_test[df_test['Client'] == 1]

# Undersamples the majority class from the prepared dataset and splits this set into a training and test set.

# takes samples from the majority class equal to the size of the minority class
df_class_0_under = df_class_0.sample(count_class_1)

# Concatenates the undersampled majority class and the minority class into a new balanced dataset
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

# Binarizes the target class to be used in the classifier
lb = LabelBinarizer()
labels_under = df_test_under['Client']

# Drops the target class from the dataset so it can't be used by the classifier to predict
df_test_under.drop(['Client'], axis=1, inplace=True)

# Splits the dataset into a test and training set, making sure the labels and data are indexed correctly
train_under, test_under, train_under_labels, test_under_labels = train_test_split(df_test_under, labels_under, 
                                                          stratify = labels_under,
                                                          test_size = 0.3, 
                                                          random_state = RSEED)

##### define functions used to evaluate the model.

def evaluate_model_under(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
#     collects baseline performance scores
    baseline = {}
    
    baseline['recall'] = recall_score(test_under_labels, [1 for _ in range(len(test_under_labels))])
    baseline['precision'] = precision_score(test_under_labels, [1 for _ in range(len(test_under_labels))])
    baseline['roc'] = 0.5
        
#     collects Random Forest performance scores on the test set
    results = {}
    
    results['recall'] = recall_score(test_under_labels, predictions)
    results['precision'] = precision_score(test_under_labels, predictions)
    results['roc'] = roc_auc_score(test_under_labels, probs)
    
#     collects Random Forest performance scores on the training set
    train_results = {}
    
    train_results['recall'] = recall_score(train_under_labels, train_predictions)
    train_results['precision'] = precision_score(train_under_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_under_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_under_labels, [1 for _ in range(len(test_under_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_under_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    
def precision_recall_threshold_under(p, r, thresholds, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    
    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    y_pred_adj = adjusted_classes(target_scores_under, t)
    print(pd.DataFrame(confusion_matrix(test_under_labels, y_pred_adj),
                       columns=['pred_Not_Client', 'pred_Client'], 
                       index=['Not_Client', 'Client']))
    
    # plot the curve
    plt.figure(figsize=(8,8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2,
             where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0.0, 1.01]);
    plt.xlim([0.0, 1.01]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    
    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
            markersize=15)
    
##### Fits the Random Forest to the undersampled dataset and prints the first set of results.

# Fits the random forest to the dataset
test_forest.fit(train_under, train_under_labels)

# Saves the predictions made by the forest in a variable and prints a confusion matrix to show precision and recall
target_scores_under = test_forest.predict(test_under.values)
print(pd.DataFrame(confusion_matrix(test_under_labels, target_scores_under),
                   columns=['pred_Not_Client', 'pred_Client'], 
                   index=['Not_Client', 'Client']))

# Collects the feature importance of each feature in a dataset and sorts them from most important to least important
fi_model = pd.DataFrame({'feature': features,
                   'importance': test_forest.feature_importances_}).\
                    sort_values('importance', ascending = False)

# Prints the full feature importance dataset
fi_model.head(-1)

##### Prints the scores and plots of the undersampled dataset, using multiple scoring methods.

# Prints the accuracy of the model on the training and test set
print("Training accuracy is: ", metrics.accuracy_score(train_under_labels, test_forest.predict(train_under)))
print("Test accuracy is: ", metrics.accuracy_score(test_under_labels, test_forest.predict(test_under)))

# Collects the predictions and probability of correctness from the Random Forest on the training data and saves them
train_rf_predictions_under = test_forest.predict(train_under)
train_rf_probs_under = test_forest.predict_proba(train_under)[:, 1]

# Collects the predictions and probability of correctness from the Random Forest on the test data and saves them
rf_predictions_under = test_forest.predict(test_under)
rf_probs_under = test_forest.predict_proba(test_under)[:, 1]

# Evaluates the model using the function defined earlier
evaluate_model_under(rf_predictions_under, rf_probs_under, train_rf_predictions_under, train_rf_probs_under)

# Collects the precision, recall and thresholds from the predictions made by the random forest
# by comparing them to the actual values, then uses an earlier defined function to plot the results
p, r, thresholds = precision_recall_curve(test_under_labels, target_scores_under)
precision_recall_threshold_under(p, r, thresholds, 0.60)

plot_precision_recall_vs_threshold(p, r, thresholds)



