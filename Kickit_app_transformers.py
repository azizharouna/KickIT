from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class Kickit_dataframe_preprocessor (BaseEstimator, TransformerMixin):
    """ 
    This is transformer class that inherits from the BaseEstimator,
    and TransformerMixin of Sklearn library. 
    The class have been modified to provide to instances 
    all the preprocessing capacity specific to the  kick dataset
    It is compatible with sklearn Pipeline and provide insights as messages when ..... 
    
    parameters =             X (required)  : the dataframe
                  choosen_axis (optional)  : a subset of features of the previous dataframe
   
   mother class = Sklearn.Base.BaseEstimator , TransformerMixin
    """

    def fit(self, X, choosen_axis = None, y=None):
        return  self  
        
    
    def transform(self, X, choosen_axis = None , y=None):       
        
        # Setting transmission Auto to 1 and manual to 0
        X['Transmission'] = X['Transmission'].replace(['AUTO'], 1)
        X['Transmission'] = X['Transmission'].replace(['MANUAL'], 0)
        X['Transmission'] = X['Transmission'].replace(['Manual'], 0)
        X['Transmission'] = X['Transmission'].replace(['?'], 0)
        # seting the list of column
        
        catList = ["Model", "Make" ,"Transmission"]
        # looping for the imputing strategy for each column
        for cat in catList:
            # replace each null value by the most frequent value
            X[cat] = X[cat].astype(str)
            X[cat]= X[cat].replace( ['?'], X[cat].mode )
            
        # seting the list of column
        numList = ["VehOdo","MMRAcquisitionAuctionAveragePrice"]
        
        # Special transition of "MMRAcquisitionAuctionAveragePrice"  to float with '?' as n
        X['MMRAcquisitionAuctionAveragePrice'] = X['MMRAcquisitionAuctionAveragePrice'].replace(['?'], '10')
        X['MMRAcquisitionAuctionAveragePrice'].astype('str')
        X['MMRAcquisitionAuctionAveragePrice'] = X['MMRAcquisitionAuctionAveragePrice'].astype('float')
        
        # looping for the imputing strategy for each column
        for num in numList:
            # replace each null value by the mean value
            X[num]= X[num].replace( ['?'], X[num].mean )
            #Converting the numerical olumns to int
            X = X[X.MMRAcquisitionAuctionAveragePrice != 0]
        
        # selecting the features to keep
        if choosen_axis == None :
            pass
        else : 
            X = X[choosen_axis]
        return  X
    
    
    
class Kickit_dataframe_balancer (BaseEstimator, TransformerMixin):
    """ 
    This is transformer class that inherits from the BaseEstimator,
    and TransformerMixin of Sklearn library. 
    The class have been modified to provide to instances 
    capacity  to balance the  kick dataset
    It is compatible with sklearn Pipeline and provide insights as messages when ..... 
    parameters =             X (required)  : the dataframe
    mother class = Sklearn.Base.BaseEstimator , TransformerMixin
    """
    def fit(self, X, y=None):
        return  self  

    def transform(self, X, y=None):
        # Count how many targets are 1 (meaning that the car is a kick)
        num_one_targets = X['IsBadBuy'].sum()
        # Set a counter for targets that are 0 
        zero_targets_counter = 0
        row_list2 =[]
        # We want to create a "balanced" dataset, so we will have to remove some input/target pairs

        for i in X.index.astype('int64'):
            if (X.loc[i,'IsBadBuy']) == 0:
                zero_targets_counter += 1
                if zero_targets_counter > num_one_targets:
                    row_list2.append(i)
        X = X.drop(index=row_list2,axis= 0)
        #resetting the index
        X.index = range(len(X.index))
        return  X
    
    
    
class Kickit_frequency_encoder (BaseEstimator, TransformerMixin):
    def fit(self, X, catCols = ["Model", "Make"],  y=None):
        return  self  
        
    def transform(self, X, catCols = ["Model", "Make"] , y=None):  
        global freq_cat_dict 
        freq_cat_dict = {}
        for col in catCols:
            df_frequency_map = X[col].value_counts().to_dict()
            X[col+'_encoded'] = X[col].map(df_frequency_map)
            freq_cat_dict.update(df_frequency_map)
        return  X  
    
    
    
    
    
class Kickit_weight_of_evidence_encoder (BaseEstimator, TransformerMixin):
    """ 
    This class provide to instances capacity  to encode the categorical features
    in the kick dataset. It is compatible with sklearn Pipeline.
    
    parameters =             X (required)  : the dataframe
    
    mother class = Sklearn.Base.BaseEstimator , TransformerMixin
    """

    global woe_cat_dict, variables
    woe_cat_dict = {}


    
    
    def fit(self, X, catCols = None, y=None):
        return  self  
    
    def transform(self, X, catCols = ["Model", "Make"], y=None):
        
        
        for col in catCols: 
            WOE = np.log((0.5 + X[col][X.IsBadBuy == 0 ].value_counts())/(0.5 + X[col][X.IsBadBuy == 1 ].value_counts()))
            WOE  = WOE.dropna()
            WOE = WOE.loc[WOE!=0]
            df_frequency_map = WOE.to_dict()
            X[col+'_encoded'] = X[col].map(df_frequency_map)
            X = X.dropna()
            woe_cat_dict.update(df_frequency_map)
            variables = X.index 
        return  X 
    

from collections import Counter
def cumulatively_categorise(column,threshold=0.75,return_categories_list=True):
    #Find the threshold value using the percentage and number of instances in the column
    threshold_value=int(threshold*len(column))
    #Initialise an empty list for our new minimised categories
    categories_list=[]
    #Initialise a variable to calculate the sum of frequencies
    s=0
    #Create a counter dictionary of the form unique_value: frequency
    counts=Counter(column)

    #Loop through the category name and its corresponding frequency after sorting the categories by descending order of frequency
    for i,j in counts.most_common():
        #Add the frequency to the global sum
        s+=dict(counts)[i]
        #Append the category name to the list
        categories_list.append(i)
        #Check if the global sum has reached the threshold value, if so break the loop
        if s>=threshold_value:
            break
        #Append the category Other to the list
        categories_list.append('Other')
        #Replace all instances not in our new categories by Other  
    new_column=column.apply(lambda x: x if x in categories_list else 'Other')

    #Return transformed column and unique values if return_categories=True
    if(return_categories_list == True):
        return new_column,categories_list
        #Return only the transformed column if return_categories=False
    else:
        return new_column


    