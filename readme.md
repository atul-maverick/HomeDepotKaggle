# Kaggle-Username   atulmaverick #


## Classification Algorithm used: K nearest neighbour mean classification ##

### Implementation details: ###
**Below are the two main steps for the relevance score calculations of test data:**
**1.	Training the relevance prediction model**

Have used the complete training data from the csv for training. 
When user assigned the relevance value to the each product based on search terms, he compared the search term with product title and image, and product description. Based on which he has assigned the relevance value to each of the search term. 

**Below are the steps for training the model:**
**1.1	Calculating the tf for test and train (Python Module: calcTfTest() )**
•	The python code calculates term frequency (used logarithmic term frequency) for each product title, description, brand name, and the corresponding search term in the training data. 
•	Perform the preprocessing using regular expressions,and string replace.
•	Store each new term into personal word list. This word list is used for spell check of search terms. 
•	Not considering document frequency, because the user examined only the particular product for assigning the values. 
•	Store the calculated tf into a dictionary based on the Product UID.
(Module calcTfTest() performs the above steps for train and test data)



**1.2	Calculating the cosine similarity of the test data (Python module: calculateCossim_train() )**
•	Considering each term frequency as vector, calculate the cosine similarity between product title and the corresponding search term, then product description and search term, and then similarly with brand name. 
•	And as the length of brand name will be smaller, the cosine similarity will have higher weightage, then product title, followed by description. 
•	Sum the cosine similarity values and store it into a dictionary with searchID as the key value. 

Each of the search ID will have cosine similarity value for which we have a relevance associated in the training data file. This forms the deciding factor for choosing the relevance value for the search IDS of test data. 

**2.	K NN classification of Relevance based on cosine similarity (Python module: classify_test() )**

•	Sort the cosine similarity dictionary based on the cosine similarity values.
•	Perform the preprocessing and spell check on the search terms before calculating the cosine similarity.
•	Calculate the cosine similarity between search term and product title, description, and brand name for each search id. (in similar way as calculated for test data).  
•	Using binary search find the index of the search ID whose cosine similarity value is closest. Get the 10 closest search ID’s.
•	The final relevance value is the mean of the relevance values of the 10 closest search ID’s.

