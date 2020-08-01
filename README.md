For this project I analyzed the correlation of pizzeria naming conventions with more expensive as well as cheaper menu prices.  

This project included: 
Data cleansing 
- Use agate to determine outliers in price columns (> 3 std deviations from the mean) 
- Removing stop words
- Drop the rows that contain duplicate restaurant names

EDA 
- Histograms of appropriate variables as well as visualization of distribution of the length of each restaurant name 
- Map visualization of the zip codes of the restaurants in my data using folium 
- Bar plot the 20 most common words in order to visualize them for better understanding 

Model
- Find the midpoint price range for each restaurant and transform into target 
- Use TFIDF-Vectorizer on restaurant names to create feature variables 
- I am using the Random Forest algorithm so that I can view which words contribute the most to the decision
- Use one-vs-rest classifier and gridsearchCV to evaluate model performance and determine the best hyperparameters  
- Train Random Forest algorithm using best hyperparameters, and use scikit-learns random forest method of .feature_importances_ to find the words that contribute most to classifying a restaurant as expensive (mid-point > $40) 
- Repeat to find words that contribute to classifying a restaurant as cheap (mid-point < $15) 
- Select Best Model from Multiple Learning Algorithms 

