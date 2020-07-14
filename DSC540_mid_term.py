# -*- coding: utf-8 -*-
"""
Holly Erickson
DSC 540 MidTerm
"""


import pandas as pd
import agate as ag



#  Replace headers, it's very simple using Pandas (Data Wrangling with Python pg. 154 – 163)

df = pd.read_csv('8358_1.csv')
# Drop currency columns (not adding any info since all in USD)
drop = ['menus.currency', 'priceRangeCurrency']
df.drop(drop, axis = 1, inplace = True)

columns = ['ID', 'Address', 'Categories', 'City', 'Country', 'Keys', 'Latitude',
       'Longitude', 'Menu Page URL', 'Max Amount', 'Min Amount',
       'Date Seen', 'Description', 'Menu Name',
       'Name', 'Postal Code', 'Price Range Min',
       'Price Range Max', 'Province']
print(df.columns) # print original headers
df.columns = columns
print(df.columns) # print replaced headers
"""
Output:
Index(['id', 'address', 'categories', 'city', 'country', 'keys', 'latitude',
       'longitude', 'menuPageURL', 'menus.amountMax', 'menus.amountMin',
       'menus.dateSeen', 'menus.description', 'menus.name', 'name',
       'postalCode', 'priceRangeMin', 'priceRangeMax', 'province'],
      dtype='object')
Index(['ID', 'Address', 'Categories', 'City', 'Country', 'Keys', 'Latitude',
       'Longitude', 'Menu Page URL', 'Max Amount', 'Min Amount', 'Date Seen',
       'Description', 'Menu Name', 'Name', 'Postal Code', 'Price Range Min',
       'Price Range Max', 'Province'],
"""
#%%
"""
Format Data to a Readable Format (Data Wrangling with Python pg. 164 – 168)
I also take care of Find Duplicates (Data Wrangling with Python pg. 175 – 178) in this step.
Pizza restaurants are listed on several rows, a row for each menu item we have data for.
We only need to see the restaurant specific info such as location once, not for each menu item. 
I have split up restaurant details and item details and will only print item details if it is a repeating restaurant.
"""
df_nan = df.copy()
df.fillna("Undefined", inplace = True)
restaurant_details = ['ID', 'Address', 'Categories', 'City', 'Country', 'Keys', 'Price Range Min', 'Price Range Max', 'Latitude', 'Longitude', 'Menu Page URL', 'Date Seen', 'Postal Code', 'Province']
item_details = ['Description',  'Min Amount', 'Max Amount']

with open('readable_pizza_file.txt', 'w', encoding='utf-8') as outfile:
    outfile.write('Pizza Restaurants. Source: https://www.kaggle.com/datafiniti/pizza-restaurants-and-the-pizza-they-sell#8358_1.csv')
    current_id = 0
    for index, row in df.iterrows():
        # iterate over each row of the dataframe and print name of row we will be looking 
        if current_id != row['ID']:
            current_id = row['ID']
            # New Restaurant, get restaurant details and menu item details
            outfile.write("\n" + "\n" + str(row['Name']))
            
             # get restaurant details
            for col in range(len(restaurant_details)):
                if str(row[restaurant_details[col]]) != "Undefined":
                    outfile.write("\n" + '{}: {}'.format(restaurant_details[col], row[restaurant_details[col]]))      
                    #outfile.write("\n" + 'Answer: {}'.format(row[restaurant_details[col]]))
            
             # get menu item details
            outfile.write("\n"  + "\n" + "Menu Item: {}".format(row['Menu Name']))
            for col in range(len(item_details)):
                if str(row[item_details[col]]) != "Undefined":
                    outfile.write("\n" + '{}: {}'.format(item_details[col], row[item_details[col]]))      
                    #outfile.write("\n" + 'Answer: {}'.format(row[item_details[col]]))
        else:
            # Same restaurant, get menu item details only
            outfile.write("\n"  + "\n" + "Menu Item: {}".format(row['Menu Name']))
            for col in range(len(item_details)):
                if str(row[item_details[col]]) != "Undefined":
                    outfile.write("\n" + '{}: {}'.format(item_details[col], row[item_details[col]]))      
                    #outfile.write("\n" + 'Answer: {}'.format(row[item_details[col]]))
        

#%%
"""
Identify outliers and bad data (Data Wrangling with Python pg. 169 – 174)
I used the library agate for this.
"""
df_orig = df_nan['Max Amount']
df_orig.dropna(inplace = True)
column_names = ['Max Amount']
column_types = [ag.Number()]


rows =[[float(val)] for val in list(df_orig.values)]
table = ag.Table(rows, column_names, column_types)

#%%
print(table.column_names)
outliers = table.stdev_outliers(column_name = 'Max Amount', deviations=3, reject=True)
print(len(outliers.rows))

outlier_list = []
for row in outliers.rows:
    outlier_list.append(float(row['Max Amount']))
    print(row['Max Amount'])

print("Mean: {}".format(table.aggregate(ag.Mean('Max Amount'))))
"""
11 Found:
116.99
116.99
116.99
118.99
100.0
116.99
312.95
310.95
311.95
312.95
69.95
Mean: 12.47918588873812754409769335
"""
df = df[~df['Max Amount'].isin(outlier_list)] # Remove outlier rows
#%%
"""
Conduct Fuzzy Matching (if you don’t have an obvious example to do this with in your
data, create categories and use Fuzzy Matching to lump data together) (Data Wrangling
with Python pg. 179 – 188)
"""
from fuzzywuzzy import fuzz
str1 = df['Name'][0]
max_match = 0
ind_match = 0

for ind, row in df.iterrows():
    str2 = row['Name']
    current_ratio = fuzz.ratio(str1, str2)
    if ind > 1 and current_ratio > max_match:
        max_match = current_ratio
        ind_match = ind

print('Closest match for {} is {} with ratio of {}.'.format(str1, df['Name'][ind_match], max_match))

"""
Output:
Closest match for Little Pizza Paradise is Little Italy Pizza Deli with ratio of 64.
"""
#%%
df.to_csv('midterm_dataset.csv')