### Merging a specific dataset
import numpy as np
import pandas as pd

# Load the datasets that need to be merged
FormData = pd.read_csv("")
FullData = pd.read_csv("")

# Merge the two datasets using the email columns from both as keys (leaves emails with typos/secondary emails)
Mergeddf = FullData.merge(FormData, how='left', left_on='E-mailadres klant', right_on='email')

# Make a list of all the emails that were properly merged
droplist = list(Mergeddf.dropna(axis=0, subset=['email'])['email'].values)

# Drop all properly merged emails from the Form dataframe
for email in droplist:
    indexnames = FormData[FormData['email'] == email].index

    FormData.drop(index=indexnames, inplace=True)

# Drop all empty rows from Form dataset
FormData = FormData.dropna(subset=['email'])

# Merge the leftover rows based on full name and birthdate
Mergeddf = Mergeddf.merge(FormData, how='left', left_on=['Voornaam klant', 'Achternaam klant', 'Geboortedatum klant'], right_on=['Naam (Voornaam)', 'Naam (Achternaam)', 'birthdate'], indicator=True)

# Due to merging the Form dataset twice its columns appear twice in the merged dataset.
# This code is used to combine these doubled up columns into one without losing the information stored in either

# Reload the Form data set in ful
FormData = pd.read_csv("")

# Save the column names for future use
ColumnList = list(FormData.columns)

# For each column name make a variable that is equal to the name of the column from the first and 2nd merge
for column in ColumnList:
    left = column + "_x"
    right = column + "_y"
    
    # Join the 2 columns while dropping empty values, resulting in all the information being in one column instead of 2
    Mergeddf[column] = Mergeddf[[left, right]].apply(
        lambda x: ''.join(x.dropna().astype(str)),
        axis=1
    )
    
    # Drop the now unnecesary columns
    Mergeddf = Mergeddf.drop([left, right], axis=1)

### Save the now complete and fixed dataset to your device as a .csv file
#Mergeddf.to_csv(r'', index = False, header=True)