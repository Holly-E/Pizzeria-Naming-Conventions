# -*- coding: utf-8 -*-
"""
Holly Erickson
Visualization
"""


import pandas as pd
import requests
import json

# Pull Data from API
key = 1 # Use Key = 1 if using for educational purposes
url = 'https://www.themealdb.com/api/json/v1/{}/random.php'.format(key)

recipes = []
for ind in range(1000):
    # I am grabbing 1000 random recipes
    response = requests.get(url)
    print(response.status_code)
    resp = json.loads(response.text)
    df = pd.DataFrame.from_dict(resp['meals']) #, orient = 'index')
    recipes.append(df)

# Storing all recipes in a dataframe
df1 = pd.concat(recipes)
#%%
#  Transform headers to a readable format (Data Wrangling with Python pg. 154 – 163)
import re
cols = df1.columns
print('Original Headers:')
print(cols)

header = ['Date Modified', 'Meal ID']
for name in cols[2:]:
    name = name[3:]
    clean = re.sub(r"([0-9]+(\.[0-9]+)?)",r" \1 ", name).strip()
    header.append(clean)

df1.columns = header
print('New Headers:')
print(df1.columns)

"""
Original Headers:
['dateModified', 'idMeal', 'strArea', 'strCategory', 'strDrinkAlternate',
       'strIngredient1', 'strIngredient10', 'strIngredient11',
       'strIngredient12', 'strIngredient13', 'strIngredient14',
       'strIngredient15', 'strIngredient16', 'strIngredient17',
       'strIngredient18', 'strIngredient19', 'strIngredient2',
       'strIngredient20', 'strIngredient3', 'strIngredient4', 'strIngredient5',
       'strIngredient6', 'strIngredient7', 'strIngredient8', 'strIngredient9',
       'strInstructions', 'strMeal', 'strMealThumb', 'strMeasure1',
       'strMeasure10', 'strMeasure11', 'strMeasure12', 'strMeasure13',
       'strMeasure14', 'strMeasure15', 'strMeasure16', 'strMeasure17',
       'strMeasure18', 'strMeasure19', 'strMeasure2', 'strMeasure20',
       'strMeasure3', 'strMeasure4', 'strMeasure5', 'strMeasure6',
       'strMeasure7', 'strMeasure8', 'strMeasure9', 'strSource', 'strTags',
       'strYoutube']

New Headers:
Index(['Date Modified', 'Meal ID', 'Area', 'Category', 'DrinkAlternate',
       'Ingredient 1', 'Ingredient 10', 'Ingredient 11', 'Ingredient 12',
       'Ingredient 13', 'Ingredient 14', 'Ingredient 15', 'Ingredient 16',
       'Ingredient 17', 'Ingredient 18', 'Ingredient 19', 'Ingredient 2',
       'Ingredient 20', 'Ingredient 3', 'Ingredient 4', 'Ingredient 5',
       'Ingredient 6', 'Ingredient 7', 'Ingredient 8', 'Ingredient 9',
       'Instructions', 'Meal', 'MealThumb', 'Measure 1', 'Measure 10',
       'Measure 11', 'Measure 12', 'Measure 13', 'Measure 14', 'Measure 15',
       'Measure 16', 'Measure 17', 'Measure 18', 'Measure 19', 'Measure 2',
       'Measure 20', 'Measure 3', 'Measure 4', 'Measure 5', 'Measure 6',
       'Measure 7', 'Measure 8', 'Measure 9', 'Source', 'Tags', 'Youtube'])
"""
#%%
# Drop columns without any data 
drop = ['Date Modified', 'DrinkAlternate']
df1.drop(drop, axis = 1, inplace = True)

#%%
# Find and remove duplicates (Data Wrangling with Python pg. 175 – 178)
print('Shape before removing duplicates:')
print(df1.shape)
df1.drop_duplicates(inplace = True)
print('Shape after removing duplicates:')
print(df1.shape)

"""
Shape before removing duplicates:
(1000, 49)
Shape after removing duplicates:
(206, 49)
"""
#%%
# Reorder columns and focus on highlighted columns
data = df1[['Meal ID', 'Meal', 'Area', 'Category', 'Ingredient 1', 'Ingredient 2', 'Ingredient 3', 'Ingredient 4', 'Ingredient 5',
       'Ingredient 6', 'Ingredient 7', 'Ingredient 8', 'Ingredient 9', 'Ingredient 10', 'Ingredient 11', 'Ingredient 12',
       'Ingredient 13', 'Ingredient 14', 'Ingredient 15', 'Ingredient 16',
       'Ingredient 17', 'Ingredient 18', 'Ingredient 19', 'Ingredient 20']]

#%%
# Replace missing values with NaN
import numpy as np
data = data.replace(r'^\s*$', 'None', regex=True)
data = data.replace(np.nan, 'None')

#%%
# Import libraries to view the distribution of ingredients in each meal using a histogram

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected=True)

import plotly.graph_objs as go

#%%
# Create a list that holds the length of each recipe
ing_len = []
for ind, row in data.iterrows():
    count = 0
    for col in range(1, 21):
        col_name = 'Ingredient {}'.format(col)
        if row[col_name] != 'None':
            #print(row[col_name])
            count += 1
    ing_len.append(count)

data['Number of Ingredients'] = ing_len
   
#%%    
# Plot the number of Ingredients by each recipe
trace = go.Histogram(
    x= ing_len,
    xbins=dict(start=0,end=90,size=1),
   marker=dict(color='#7CFDF0'),
    opacity=0.75)
list_trace = [trace]
layout = go.Layout(
    title='Distribution of # of Ingredients',
    xaxis=dict(title='Number of Ingredients'),
    yaxis=dict(title='Count of Recipes'),
    bargap=0.1,
    bargroupgap=0.2)

fig = go.Figure(data=list_trace, layout=layout)
plot(fig)       

#%%
# Transform words in string columns to title case
copy = data.copy()
for col in copy.columns[1:-1]:
    copy[col].apply(str)
    data[col] = copy[col].str.title()
#%%
data.to_csv('final_dataset.csv')