from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np


# data cleaning
# create a Year-Month string list for data retrieval
def file_date():
    date_list = []
    for year in [2022, 2023, 2024]:
        month_range = 13 if year != 2024 else 11
        for month in range(1, month_range):
            date = f"{year}-0{month}" if month < 10 else f"{year}-{month}"
            date_list.append(date)
    return date_list


# data loading, cleaning, statistics and itegrate in a general dataset
def process_data(date, final_df):
    # load the data
    file_path = f"{date}-north-yorkshire-street.csv"
    try:
        crime = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return final_df

    # data cleaning：deleting rows with null "LSOA code" data
    crime_clean = crime.dropna(subset=["LSOA code"])

    # load template file
    zero = pd.read_csv("zero.csv")

    # Statistics: crime cases within each LSOA
    for index, row in crime_clean.iterrows():
        crime_site = row["LSOA code"]
        zero.loc[zero["LSOA"] == crime_site, "Crime"] += 1

    # add the date column
    zero['Month'] = date

    # add the statistics data in the final dataset
    final_df = pd.concat([final_df, zero], ignore_index=True)

    # return the itegrate dataset
    return final_df


# access Year-Month string list
date_list = file_date()

# initialize the final itegrate dataset
crime_data = pd.DataFrame()

# iterate through each date and itegrate into the final dataset
for date in date_list:
    crime_data = process_data(date, crime_data)

# save the final dataset as a backup
crime_data.to_csv("crime.csv", index=False)

# check null
# calculate the rows with null value
null_rows_count = crime_data["LSOA"].isnull().sum()
# calculate the rows without null value
not_null_rows_count = crime_data["LSOA"].notnull().sum()
# visualization for data cleaning
data_cleaning_visualization = pd.DataFrame({
    'Category': ['Null Values', 'Non-null Values'],
    'Count': [null_rows_count, not_null_rows_count]
})
# draw the bar plot
plt.bar(data_cleaning_visualization['Category'], data_cleaning_visualization['Count'], width=0.2,
        color=['Yellow', 'Blue'])
# add title and labels
plt.title('Count of Null and Non-Null Values in "LSOA code"')
plt.xlabel('Category')
plt.ylabel('Count')

# show the plot
plt.show()

# format check
# show format of the month column
print(crime_data["Month"].unique())

# data cleaning of PPFI
# Filter the crime data of 2022-10
crime_data_2022_10 = crime_data.loc[crime_data["Month"] == '2022-10']

# load the PPFI data
ppfi = pd.read_csv("PPFI.csv")
# merge the PPFI data with crime_data_2022_10
crime_ppfi_merge = pd.merge(crime_data_2022_10, ppfi, how="left")
# Filter the PPFI data from the merged data
ppfi_data = crime_ppfi_merge[['LSOA', 'pp_dec_domain_supermarket_proximity', 'pp_dec_domain_supermarket_accessibility',
                              'pp_dec_domain_ecommerce_access', 'pp_dec_domain_socio_demographic',
                              'pp_dec_domain_nonsupermarket_proximity', 'pp_dec_domain_food_for_families',
                              'pp_dec_domain_fuel_poverty']]

# calculate the rows with null value
null_rows_count = ppfi_data.isnull().any(axis=1).sum()

# calculate the rows without null value
not_null_rows_count = ppfi_data.notnull().all(axis=1).sum()

# visualization for data cleaning
counts = pd.DataFrame({
    'Type': ['Null Rows', 'Non-Null Rows'],
    'Count': [null_rows_count, not_null_rows_count]
})

# draw the bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x='Type', y='Count', data=counts, palette='viridis')
plt.title('Count of Null and Non-Null Rows')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()

# clean the null rows
ppfi_data = ppfi_data.dropna()

# exploratory visualization of crime data
# create a figure window, and set up the subplot layout (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# draw the bar plot
axes[0].bar(crime_data_2022_10['LSOA'], crime_data_2022_10['Crime'], color='lightblue')
axes[0].set_title('Bar Chart of Crime by LSOA')
axes[0].set_xlabel('LSOA')
axes[0].set_ylabel('Crime')
axes[0].set_xticks([])  # hide the x label

# draw the boxplot
boxplot_dict = axes[1].boxplot(crime_data_2022_10['Crime'], patch_artist=True)

# set the styles of the boxplot
for box in boxplot_dict['boxes']:
    box.set(facecolor='lightblue', edgecolor='grey', linewidth=1)

# set the styles of medians
for median in boxplot_dict['medians']:
    median.set(color='firebrick', linewidth=0.5)

# set the styles of outliers
for flier in boxplot_dict['fliers']:
    flier.set(marker='o', color='r', markersize=2, linewidth=0.5)

# set the styles of means
meanprops = dict(marker='D', markerfacecolor='indianred', markersize=5, markeredgecolor='black')
axes[1].plot([1], [crime_data_2022_10['Crime'].mean()], **meanprops)  # add the mean marker
axes[1].set_title('Boxplot of Crime Activity')
axes[1].set_ylabel('Crime Value')
axes[1].set_xticks([1])
axes[1].set_xticklabels(['Crime'])

# adjust the subplots
plt.tight_layout()
plt.show()

# exploratory visualization of FFPI data
# line chats
# FFPI data

# create a figure window, and set up the subplot layout (3 rows, 3 columns)
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Iterate through each variable and plot a line chart.
for i, column in enumerate(ppfi_data.columns[1:]):  # pass the "Month" column
    ax = axes[i // 3, i % 3]  # settle the location of subplot
    ax.bar(ppfi_data['LSOA'], ppfi_data[column], color='#65A8B0')
    ax.set_title(column)
    ax.set_xlabel('LSOA')
    ax.set_ylabel('Value')
    ax.set_xticks([])

# hide the last two subplot（only 7 variables existing）
axes[2, 1].axis('off')
axes[2, 2].axis('off')

# adjust the subplot
plt.tight_layout()
plt.show()

# boxplot
ppfi_data_boxplot = ppfi_data.set_index(ppfi_data.columns[0])
# print(ppfi_data_boxplot.iloc[2])
sns.boxplot(data=ppfi_data_boxplot, palette='viridis')

# show the plots
# set the title and labels
plt.title('Boxplot of Multiple Variables')
plt.xlabel('Variable')
plt.ylabel('Value')
short_x_labels = ["Supermarket proximity", "Supermarket accessibility", "Ecommerce access", "Socio demographic",
                  "Non-supermarket proximity", "Food for families", "Fuel poverty"]
plt.xticks(range(len(short_x_labels)), short_x_labels, rotation=45, ha='right')
plt.show()

# Statistics Modelling
# merge the crime and PPFI data for the statistics analysis
crime_ppfi_reg = pd.merge(crime_data, ppfi_data, how='left')
# print(crime_ppfi_reg.iloc[2])
# remove the 'LSOA' and 'Month' columns
crime_ppfi_reg = crime_ppfi_reg.drop(columns=["LSOA", "Month"])
crime_ppfi_reg = crime_ppfi_reg.dropna()
# print(crime_ppfi_reg.iloc[2])
# calculate the correlation matrix
corr_matrix = crime_ppfi_reg.corr()
# create a mask for the heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# show the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

# correlation analysis: crime activity data (dependent) -- socio-demographic index (independent)
# independent variable
x = crime_ppfi_reg[["pp_dec_domain_socio_demographic"]].values
# dependent variable
y = crime_ppfi_reg[["Crime"]].values
# divide the dataset into training sets and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
# establish the linear regression model
reg_model = LinearRegression()
# fit the model
reg_model.fit(x_train, y_train)

# access the slope and intercept
slope = reg_model.coef_[0]
intercept = reg_model.intercept_

# prediction
y_prediction = reg_model.predict(x_test)
# evaluate model performance
mse = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)

# show the scatter plot
plt.scatter(x, y, color=(29 / 255, 103 / 255, 172 / 255), label='real data', s=5)
plt.plot(x, reg_model.predict(x), color=(250 / 255, 196 / 255, 127 / 255), label='fitted line')
plt.title('Result of Linear Regression Analysis')
plt.xlabel("Socio-demographic Index")
plt.ylabel('Crime Activity')
plt.legend()
plt.show()

# Data Visualization
# LSTM Prediction Model
# load the data
data = pd.read_csv("crime.csv")

# pivot table
data_pivot = data.pivot_table(index="Month", columns="LSOA", values="Crime")

# Data Normalization
scaler = MinMaxScaler()
scaler_data = scaler.fit_transform(data_pivot)


# create a dataset that LSTM model can access
# param data: pivot data (normalized)
# param timestep: use how many months data for the prediction
def create_dataset(data, time_steps=1):
    # x: input data (sample quantity, timestep, LSOA quantity)
    # y: output data ()
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(x), np.array(y)


# set the time steps
time_steps = 12
# use 12 months data for prediction
x, y = create_dataset(scaler_data, time_steps)

# divide the dataset into training set and testing set
# shuffle=False: keep the data as time-series permutation instead of a random one
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)

# reshape input as [samples, time steps, features] (feature is the quantity of LSOA)
n_features = data_pivot.shape[1]
# reshape training set
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], n_features))
# reshape testing set
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], n_features))

# establish the LSTM model
lstm_model = tf.keras.Sequential([
    # the first layer of LSTM with 50 neurons units
    tf.keras.layers.LSTM(units=64, activation='tanh', input_shape=(time_steps, n_features), return_sequences=True),
    # the dropout layer: drop out 20% units randomly to prevent overfitting
    tf.keras.layers.Dropout(0.2),
    # the second layer of LSTM with 35 neurons units
    tf.keras.layers.LSTM(units=32, activation='tanh'),
    # the dropout layer: drop out 20% units randomly to prevent overfitting
    tf.keras.layers.Dropout(0.2),
    # output the values of prediction
    tf.keras.layers.Dense(units=n_features)  # output each prediction value within LSOA
])

# compile the model
lstm_model.compile(optimizer='adam', loss='mse')

# print the model architecture
lstm_model.summary()

# train the model
# add EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = lstm_model.fit(x_train, y_train, epochs=45, batch_size=32, validation_split=0.1, verbose=1,
                         callbacks=[early_stopping])

# visualize the training loss and validation loss
plt.plot(history.history['loss'], label='training loss', color=(29 / 255, 103 / 255, 172 / 255), linestyle="--")
plt.plot(history.history['val_loss'], label='validation loss', color=(250 / 255, 196 / 255, 127 / 255))
plt.title('Model Training Loss')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.legend()
plt.show()

# evaluate the model
loss = lstm_model.evaluate(x_test, y_test, verbose=0)
print(f'validation loss: {loss}')

# prediction
predictions = lstm_model.predict(x_test)

# Inverse Normalization
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# select the reality and prediction data of step 1
step_1_index = 0  # index of step 1
y_test_step_1 = y_test[step_1_index, :]  # real data
predictions_step_1 = predictions[step_1_index, :]  # prediction data
# create a new dataframe including LSOA and crime activity
lsoa_crime_df = pd.DataFrame({
    'LSOA': data_pivot.columns,
    'Real Crime': y_test_step_1,
    'Predicted Crime': predictions_step_1
})

# scatter plot
plt.figure(figsize=(12, 6))

# 绘制真实犯罪数量的折线图
plt.plot(lsoa_crime_df['LSOA'], lsoa_crime_df['Real Crime'], label='Real Crime', marker='o', linewidth=1,
         linestyle="--")

# 绘制预测犯罪数量的折线图
plt.plot(lsoa_crime_df['LSOA'], lsoa_crime_df['Predicted Crime'], label='Predicted Crime', marker='.', linewidth=1)

plt.title('Crime Prediction for Step 1')
plt.xlabel('LSOA')
plt.ylabel('Crime')
plt.xticks([])
plt.legend()
plt.tight_layout()  # adjust subplot index automatically
plt.show()

# Future Prediction
# future prediction
# model: trained model
# last_sequence: the last time-series clip shape as (time_steps, LSOA quantity)
# n_step: how many months of data to predict
def prediction_future(model, last_sequence, n_steps):
    # forecast: crime activities in the later "n_step" months shape as (time_steps, LSOA quantity)
    forecast = []
    for _ in range(n_steps):
        # reshape the last time-series as LSTM input format
        x_input = np.array(last_sequence).reshape((1, time_steps, n_features))
        # predict the crime activity in next month
        yhat = model.predict(x_input, verbose=0)
        # add the prediction result in the forecast list
        forecast.append(yhat[0])
        # update last_sequence, remove the data of the first month and add the prediction data in next month
        last_sequence = np.concatenate((last_sequence[1:], yhat[0].reshape(1, n_features)), axis=0)
    return np.array(forecast)


n_future_steps = 1  # predict the data of the next month
last_sequence = x_test[-1]  # use the last sequence as the beginning step

# prediction
future_forecast = prediction_future(lstm_model, last_sequence, n_future_steps)

# inverse normalize
future_forecast = scaler.inverse_transform(future_forecast)

# create future date index
last_date = data_pivot.index[-1]
future_date = pd.date_range(start=last_date, periods=n_future_steps + 1, freq='M')[1:]
# print(future_forecast, future_date)

# Spatial Visualization
# assemble dataset for final data visualization
# use the template file to built the dataset
vis_lsoa = pd.read_csv("zero.csv")
# add the predicted crime data
vis_crime = pd.DataFrame(future_forecast[0], columns=["Crime"])
vis_lsoa["Crime"] = vis_crime["Crime"]
# add the PPFI data
vis_data = pd.merge(vis_lsoa, ppfi_data, how="left")
# drop useless columns
vis_data = vis_data.drop(columns=["pp_dec_domain_supermarket_proximity", "pp_dec_domain_supermarket_accessibility",
                                  "pp_dec_domain_ecommerce_access", "pp_dec_domain_nonsupermarket_proximity",
                                  "pp_dec_domain_food_for_families", "pp_dec_domain_fuel_poverty"])

# load geojson file
gdf = gpd.read_file("north_yorkshire.geojson")

# change column name before merge
gdf = gdf.rename(columns={"LSOA21CD": "LSOA"})

# merge datasets
gdf = gdf.merge(vis_data, on="LSOA")

# convert polygon to points (PPFI visualization)
centroids = gdf.centroid
gdf_1 = gpd.GeoDataFrame(geometry=centroids, crs=gdf.crs)
gdf_1["LSOA"] = gdf["LSOA"]
gdf_1["pp_dec_domain_socio_demographic"] = gdf["pp_dec_domain_socio_demographic"]


class NorthArrow(FancyArrowPatch):
    def __init__(self, ax, **kwargs):
        super().__init__((0.1, 0.1), (0.2, 0.1), **kwargs)
        self.set_transform(ax.transAxes)
        ax.add_patch(self)


# add plot figure
fig, ax = plt.subplots(figsize=(15, 15))

# plot the polygon data
gdf.plot(ax=ax, column='Crime', cmap='cividis_r', alpha=0.5, edgecolor='black')

# plot the scatter data
gdf_1.plot(ax=ax, column='pp_dec_domain_socio_demographic', cmap='cividis_r', markersize=5, legend=True)

# add north arrow
north_arrow = NorthArrow(ax, arrowstyle='-|>', color='k', shrinkA=5, shrinkB=5)

# add legend and title
plt.legend(loc='upper left', fontsize=20)
plt.title('Crime Rate and Socio-Demographic Visualization')
plt.show()