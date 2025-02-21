# Importing needed libraries and verifying with a print function
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

print('Modules are imported.')

# Ensuring all columns are printed in output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Reading the CSV file and storing it in a variable
corona_dataset_csv = pd.read_csv('covid19_Confirmed_dataset.csv')

# Printing the first 5 rows of the dataset
print(corona_dataset_csv.head())

# Checking the shape of the dataframe and outputting the number of rows and columns
dataframeShape = corona_dataset_csv.shape
print(f"{dataframeShape[0]} rows, {dataframeShape[1]} columns")

# Dropping Lat and Long columns (not needed for analysis)
corona_dataset_csv.drop(['Lat', 'Long'], axis=1, inplace=True)

print('\nCorona dataset without Lat/Long Columns')
print(corona_dataset_csv.head(10))

# Aggregating rows by country
corona_dataset_aggregated = corona_dataset_csv.groupby("Country/Region").sum()
print('\nDataset aggregated by Country/Regions')
print(corona_dataset_aggregated.head(10))

# Ensuring numeric columns are correctly formatted
corona_dataset_aggregated = corona_dataset_aggregated.apply(pd.to_numeric, errors='coerce')

# Visualizing data using a graph for three selected countries
plt.figure(figsize=(10, 6))
corona_dataset_aggregated.loc['China'].plot(label="China")
corona_dataset_aggregated.loc['Italy'].plot(label="Italy")
corona_dataset_aggregated.loc['Spain'].plot(label="Spain")

plt.legend()
plt.xlabel("Days")
plt.ylabel("Total Confirmed Cases")
plt.title("Covid-19 Confirmed Cases Over Time")
plt.show()

# Visualizing the first derivative (rate of change of infections) for China
plt.figure(figsize=(10, 6))
corona_dataset_aggregated.loc['China'].diff().plot()
plt.xlabel("Days")
plt.ylabel("Daily Increase in Cases")
plt.title("Rate of Change of Infections in China")
plt.show()

# Finding the maximum rate of infection for selected countries
print("Maximum rate of infection for China:", corona_dataset_aggregated.loc['China'].diff().max())
print("Maximum rate of infection for Italy:", corona_dataset_aggregated.loc['Italy'].diff().max())
print("Maximum rate of infection for Spain:", corona_dataset_aggregated.loc['Spain'].diff().max())

# Finding the maximum rate of infection for all countries
countries = list(corona_dataset_aggregated.index)
max_infection_rates = [corona_dataset_aggregated.loc[country].diff().max() for country in countries]

# Creating a new column 'max infection rate' in corona_dataset_aggregated
corona_dataset_aggregated['max infection rate'] = max_infection_rates

# Creating a new DataFrame with only country names and max infection rates
corona_data = pd.DataFrame(corona_dataset_aggregated['max infection rate'])
print(corona_data.head())

# Importing world happiness report dataset
world_happiness_report = pd.read_csv("worldwide_happiness_report.csv")
print(world_happiness_report.head())

# Dropping unnecessary columns
columns_to_drop = ['Overall rank', 'Score', 'Generosity', 'Perceptions of corruption']
world_happiness_report.drop(columns=columns_to_drop, axis=1, inplace=True)

# Setting country names as index
world_happiness_report.set_index('Country or region', inplace=True)

# Joining the two datasets
print("Corona Dataset")
print(corona_data.head())

print("World Happiness Dataset")
print(world_happiness_report.head())

data = world_happiness_report.join(corona_data).copy()

print("Corona Data vs World Happiness Data")
print(data.head())

# Plotting GDP per capita vs maximum infection rate
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['GDP per capita'], y=np.log(data['max infection rate']))
plt.xlabel("GDP per capita")
plt.ylabel("Log of Max Infection Rate")
plt.title("GDP per Capita vs Maximum Infection Rate")
plt.show()

# Regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x=data['GDP per capita'], y=np.log(data['max infection rate']))
plt.xlabel("GDP per capita")
plt.ylabel("Log of Max Infection Rate")
plt.title("GDP per Capita vs Maximum Infection Rate (Regression)")
plt.show()


# Plotting Freedom to Make Life Choices vs Maximum Infection Rate
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Freedom to make life choices'], y=np.log(data['max infection rate']))
plt.xlabel("Freedom to Make Life Choices")
plt.ylabel("Log of Max Infection Rate")
plt.title("Freedom to Make Life Choices vs Maximum Infection Rate")
plt.show()

# Regression plot for Freedom to Make Life Choices vs Maximum Infection Rate
plt.figure(figsize=(10, 6))
sns.regplot(x=data['Freedom to make life choices'], y=np.log(data['max infection rate']))
plt.xlabel("Freedom to Make Life Choices")
plt.ylabel("Log of Max Infection Rate")
plt.title("Freedom to Make Life Choices vs Maximum Infection Rate (Regression)")
plt.show()

# Reset the index to convert country names from index to a column
corona_data_reset = corona_data.reset_index()
corona_data_reset.rename(columns={"Country/Region": "Country"}, inplace=True)

# Create an interactive world map
fig = px.choropleth(
    corona_data_reset,
    locations="Country",
    locationmode="country names",
    color="max infection rate",
    title="Global COVID-19 Cases (Max Infection Rate)",
    color_continuous_scale="Reds",
)

fig.show()
