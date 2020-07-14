# -*- coding: utf-8 -*-
"""
Holly Erickson
DSC550 Exercise 8.3 
Original Analysis Case Study Part 3
"""

import pandas as pd
import matplotlib.pyplot as plt

"""
Create Part 3 of your Analysis Case Study project. 
Part 3 should consist of Model Evaluation and Selection.
You can use any methods/tools you think are most appropriate.
Write the step-by-step instructions for completing the Model Evaluation and Selection part of your case study.
"""

#Step 1:  Load data into a dataframe
addr1 = "8358_1.csv"
data = pd.read_csv(addr1)
#%%
"""
# Step 2:  check the dimension of the table
print("The dimension of the table is: ", data.shape)

#Step 3:  Examine the variables and their types
print(data.head(5))

print("Describe Data")
print(data.describe())
print("Summarized Data")
print(data.describe(include=['O']))


#Step 4: draw histograms of appropriate variables
print(data.columns)
target_cols = ['menus.amountMax', 'menus.amountMin', 'priceRangeMin','priceRangeMax']

#%%
#Step 5: View zipcodes using Folium
import folium
# Read in our map:
my_USA_map = './DSC550/folium-master/folium-master/examples/data/us-states.json'
map = folium.Map(location=[48, -102], zoom_start=3)

for index, row in data.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=row['name'],
        icon=folium.Icon(size=(5,5))
    ).add_to(map)

# Create interactive map and save as html
map
map.save("mymap.html")

#%%
#Step 6: Identify outliers using the library agate.
import agate as ag
import agatestats

# Loop through each column of our relevant columns
for col in target_cols:
    df_outlier = data[col]
    df_outlier.dropna(inplace = True)
    column_names = [col]
    column_types = [ag.Number()]
    
    # create agate table using column data
    rows =[[float(val)] for val in list(df_outlier.values)]
    table = ag.Table(rows, column_names, column_types)
    
    # Print mean and outliers > 3 std deviations from mean
    print(table.column_names)
    outliers = table.stdev_outliers(column_name = col, deviations=3, reject=True)
    print(col, ": ", len(outliers.rows), "Outliers")
    
    outlier_list = []
    print("Mean", table.aggregate(ag.Mean(col)))
    for row in outliers.rows:
       outlier_list.append(float(row[col]))
       print(row[col])


#%%
# Step 7 View the distribution of number of words in each restaurant name using a histogram
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 
init_notebook_mode(connected=True)

import plotly.graph_objs as go
all_listings = list(data['name'])
unique_listing = list(set(all_listings))
# Create a list that holds the length of each restaurant name
name_len = []
for restaurant in unique_listing:
    name_len.append(len(restaurant))
    
trace = go.Histogram(
    x= name_len,
    xbins=dict(start=0,end=90,size=1),
   marker=dict(color='#7CFDF0'),
    opacity=0.75)
list_trace = [trace]
layout = go.Layout(
    title='Distribution of Name Length',
    xaxis=dict(title='Words in Name'),
    yaxis=dict(title='Count of Restaurants'),
    bargap=0.1,
    bargroupgap=0.2)

fig = go.Figure(data=list_trace, layout=layout)
plot(fig)       

#%%
# Step 8: View the most commonly used words in the restaurant names
from collections import Counter
# this list stores all the words in all restaurants (with duplicates)
all_words = [] 
for full_name in unique_listing:
    name_split = full_name.split()
    for word in name_split:
        all_words.append(word) 

# Count how many times each word occurs
countingr = Counter()
for word in filtered_words:
     countingr[word] += 1
     
print("The most commonly used words (with counts) are:")
print("\n")
print(countingr.most_common(20))
print("\n")
print("The number of unique words in our training sample is {}.".format(len(countingr)))

#%%
# Step 9: Bar plot the 20 most common words in order to visualize them for better understanding
mostcommon = countingr.most_common(20)
mostcommoningr = [i[0] for i in mostcommon]
mostcommoningr_count = [i[1] for i in mostcommon]

trace = go.Bar(
            x=mostcommoningr_count[::-1],
            y= mostcommoningr[::-1],
            orientation = 'h',marker = dict(),
)
layout = go.Layout(
    xaxis = dict(title= 'Number of occurences in all restaurant names', ),
    yaxis = dict(title='Word',),
    title= '20 Most Common Words', titlefont = dict(size = 20),
    margin=dict(l=150,r=10,b=60,t=60,pad=5),
    width=800, height=500, 
)
trace_data = [trace]
fig = go.Figure(data=trace_data, layout=layout)
plot(fig, filename='horizontal-bar')
"""
#%%
# Step 10: Drop the rows that contain duplicate restaurant names
print(data.columns)

drop_list = ['address', 'categories', 'city', 'country', 'keys', 'latitude',
       'longitude', 'menuPageURL', 'menus.amountMax', 'menus.amountMin',
       'menus.currency', 'menus.dateSeen', 'menus.description', 'menus.name',
       'postalCode', 'priceRangeCurrency', 'province']
copy = data.copy()
copy.drop(drop_list, axis = 1, inplace = True)
# dropping duplicte values 
copy.drop_duplicates(inplace = True) 
#%%
# Step 11: Find the midpoint price range for each restaurant and transform into target
# Add a column for mid-price of restaurant 
import numpy as np
mid = []
name = []
for ind, row in copy.iterrows():
    high = row['priceRangeMax']
    low = row['priceRangeMin']
    middle = ((high-low)/2) + low
    if np.isnan(middle) == False:
        mid.append(middle)
        name.append(row['name'])
        
df = pd.DataFrame()
df['mid'] = mid
df['name'] = name
df['target'] = np.nan
# Replace mid with category 
df.loc[df['mid'] >= 40, 'target'] = 1 # REDO FOR 40
df.loc[df['mid'] < 40, 'target'] = 0

print("Target Value Counts: ", df.target.value_counts())
Y = df['target']
#%%
# Step 12: Use TFIDF-Vectorizer on restaurant names to create feature variables
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.preprocessing import LabelEncoder

# Feature Engineering 
tfidf = TfidfVectorizer(binary=True)
def tfidf_features(txt, flag):
    if flag == "train":
        x = tfidf.fit_transform(txt)
    else:
	    x = tfidf.transform(txt)
    x = x.astype('float16')
    return x 

X = tfidf_features(df['name'], flag="train")


#%%
# Step 13:  Use one-vs-rest classifier and gridsearchCV to evaluate model performance and determine the best hyperparameters 

from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(class_weight = "balanced", n_estimators=200, min_samples_split= 10, min_samples_leaf=4, n_jobs=-1) # Modified for Pass 11

"""
SVC(C=100, # penalty parameter
	 			 kernel='rbf', # kernel type, rbf working fine here
	 			 degree=3, # default value
	 			 gamma=1, # kernel coefficient
	 			 coef0=1, # change to 1 from default value of 0.0
	 			 shrinking=True, # using shrinking heuristics
	 			 tol=0.001, # stopping criterion tolerance 
	      		 probability=False, # no need to enable probability estimates
	      		 cache_size=200, # 200 MB cache size
	      		 class_weight=None, # all classes are treated equally 
	      		 verbose=False, # print the logs 
	      		 max_iter=-1, # no limit, let it run
          		 decision_function_shape=None, # will use one vs rest explicitly 
          		 random_state=None)
"""
model = OneVsRestClassifier(classifier, n_jobs=4)

## Model Tuning 
parameters = {"estimator__min_samples_split": [4,5,10], "estimator__min_samples_leaf": [2, 4, 10] }
#{"estimator__gamma":[0.01, 0.5, 0.1, 2, 5, 10]}
grid_search = GridSearchCV(model, param_grid=parameters)
# This will tell you how the parameters are being passed incase you get errors in your grid_search
#for param in grid_search.get_params().keys(): 
#    print(param)
    
grid_search.fit(X, Y) 
print ("Best Score: ", grid_search.best_score_)
print("Best Params: ", grid_search.best_params_)


#%%
# Step 14: Train Random Forest algorithm using best hyperparameters
# Use scikit-learns random forest method of .feature_importances_ to find the words that contribute most to classifying a restaurant price-point
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

clf = RandomForestClassifier( class_weight = "balanced", n_estimators=200, min_samples_split= 10, min_samples_leaf=2, n_jobs=-1) # Modified for Pass 11
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
score = clf.score(x_test, y_test)
print("Score: ", score)
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)
#print(clf.feature_importances_)

clf.fit(X, Y)
feat = pd.DataFrame()
feat['importance'] = clf.feature_importances_
feat['feat'] = tfidf.get_feature_names()
