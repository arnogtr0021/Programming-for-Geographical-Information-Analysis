# Background of the Study
With the improvement of police databases and machine learning technology, it is much easier for the public service to analyze the crime status and predict its development.
# Aims of the Study

This study is aimed at researching the intrinsic relationship between crime activity and living standards. The possibility of utilizing machine learning model in prediction of crime activity will also be explored. 

The research is primarily helpful for the policy makers and police department. The policy makers could realize the main criteria of living standards that effect the crime activity and release appropriate policy on living standard of the residence to reduce the crime activity. The police department could take advantage of the prediction model to predict the potential crime activity in different areas in the future. It would help the police department to distribute their force and resource efficiently, and enforce their response to the emergency cases.

# Objectives

The data of crime activity and living standard will be utilized in the research to demonstrate their relationship. A correlation and regression analysis will be conducted to find the factor of the living standard that is most relevant to the crime activities.
	
 A Long Short-Term model will be conducted for crime activity prediction. The time-series crime activity data will be introduced to train the model. Once the model is evaluated to be accurate enough for the prediction, The prediction of the next few months and the most relevant living standard factor will be introduced in the data visualization step.
 
The study area is limited in North Yorkshire of England and the study is conducted at the Lower Super Output Area (LSOA) level.

# Data Resource

There are three different kinds of data involved in the study. Two kinds of statistics data: the crime data (01/2022-10/2024) and Priority Places for Food Index data (10/2022), and one spatial data: LSOA boundaries of North Yorkshire.

The crime data is collected from: https://data.police.uk/data. The datasets record the crime cases reported each month. The original crime data will be statistically analysed and converted to a format that shows the quantity of crimes happened in LSOA within a month.

The Priority Places for Food Index (PPFI) data is collected from: https://data.cdrc.ac.uk/dataset. The dataset records the multiple indices that reflects the living standards. As the dataset includes the data in the entire UK range, the data of North Yorkshire will be filtered for the regression analysis and visualization.

The LSOA boundaries data is collected from: https://www.data.gov.uk/dataset. The dataset will be utilized in the spatial visualization step. As the data is recorded as shapefile, it will be converted to GeoJSON format before visualization operation.

# Code Function
## Data Cleaning
Select the crime data and PPFI data in the range of North Yorkshire from the national width and remove the rows with null value in the datasets.
## Exploratory Data Description
An important step before data analysis, which can help the researchers to understand the characteristics of the data. A series of bar plots and boxplots are conducted to show the distribution status of the two datasets.
## Linear Regression Analysis
The analysis to study the relationship between crime activities and different factors in North Yorkshire. A correlation analysis is ahead of the regression analysis to find the index that have the highest correlation with crime activity for the following analysis. The result demonstrates the relationship between the index and crime activity.
## Long Short-term Model 
The machine learning model is utilized in the study. With the support of time-series crime data. The model is trained to predict the crime density distribution in North Yorkshire at a LSOA level. The model is adjusted and tested multiple times to keep the predicted results at a high precision level.
## Data Visualization
Visualize the result of the prediction model and demonstrate the spatial distribution status of two different kinds of data.

