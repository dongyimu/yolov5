# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T01:52:11.833868Z","iopub.execute_input":"2024-08-06T01:52:11.834267Z","iopub.status.idle":"2024-08-06T01:52:14.225598Z","shell.execute_reply.started":"2024-08-06T01:52:11.834235Z","shell.execute_reply":"2024-08-06T01:52:14.224358Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np
import pandas as pd
import random
import datetime as dt
import seaborn as sns
from re import search
 
random.seed(333)
pd.options.mode.chained_assignment = None
 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T01:52:14.228037Z","iopub.execute_input":"2024-08-06T01:52:14.228680Z","iopub.status.idle":"2024-08-06T01:52:18.784556Z","shell.execute_reply.started":"2024-08-06T01:52:14.228635Z","shell.execute_reply":"2024-08-06T01:52:18.783363Z"},"jupyter":{"outputs_hidden":false}}
# Read train/test data and check colnames & NA's:
 
original_train = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv')
original_test = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv')
 
# Dataframe info:
print(original_train.info())
 
# Check NAs:
original_train.isna().any()

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:08:56.666561Z","iopub.execute_input":"2024-08-06T02:08:56.666945Z","iopub.status.idle":"2024-08-06T02:08:56.692486Z","shell.execute_reply.started":"2024-08-06T02:08:56.666915Z","shell.execute_reply":"2024-08-06T02:08:56.691269Z"},"jupyter":{"outputs_hidden":false}}
original_train['store_nbr'].unique().__len__() # 54 stores

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:08:58.255846Z","iopub.execute_input":"2024-08-06T02:08:58.256301Z","iopub.status.idle":"2024-08-06T02:08:58.534244Z","shell.execute_reply.started":"2024-08-06T02:08:58.256262Z","shell.execute_reply":"2024-08-06T02:08:58.533094Z"},"jupyter":{"outputs_hidden":false}}
original_train['family'].unique().__len__() # 33 products

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:08:59.927093Z","iopub.execute_input":"2024-08-06T02:08:59.927509Z","iopub.status.idle":"2024-08-06T02:08:59.934662Z","shell.execute_reply.started":"2024-08-06T02:08:59.927474Z","shell.execute_reply":"2024-08-06T02:08:59.933470Z"},"jupyter":{"outputs_hidden":false}}
 len(original_train) / 54 / 33 # 1684 days (between 4 and 5 years)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:09:00.974207Z","iopub.execute_input":"2024-08-06T02:09:00.974731Z","iopub.status.idle":"2024-08-06T02:09:00.982881Z","shell.execute_reply.started":"2024-08-06T02:09:00.974696Z","shell.execute_reply":"2024-08-06T02:09:00.981387Z"},"jupyter":{"outputs_hidden":false}}
original_train['date'].iloc[0] # 2013-01-01 is start

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:09:02.190893Z","iopub.execute_input":"2024-08-06T02:09:02.191927Z","iopub.status.idle":"2024-08-06T02:09:02.199132Z","shell.execute_reply.started":"2024-08-06T02:09:02.191889Z","shell.execute_reply":"2024-08-06T02:09:02.198052Z"},"jupyter":{"outputs_hidden":false}}
original_train['date'].iloc[-1] # 2017-08-15 is end

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:09:05.611522Z","iopub.execute_input":"2024-08-06T02:09:05.611987Z","iopub.status.idle":"2024-08-06T02:09:05.619127Z","shell.execute_reply.started":"2024-08-06T02:09:05.611953Z","shell.execute_reply":"2024-08-06T02:09:05.617952Z"},"jupyter":{"outputs_hidden":false}}
 len(original_test) / 54 / 33 # 16 days

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:09:06.903554Z","iopub.execute_input":"2024-08-06T02:09:06.903991Z","iopub.status.idle":"2024-08-06T02:09:06.911467Z","shell.execute_reply.started":"2024-08-06T02:09:06.903957Z","shell.execute_reply":"2024-08-06T02:09:06.910264Z"},"jupyter":{"outputs_hidden":false}}
original_test['date'].iloc[0] # 2017-08-16 is test start

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:09:07.927785Z","iopub.execute_input":"2024-08-06T02:09:07.928182Z","iopub.status.idle":"2024-08-06T02:09:07.935895Z","shell.execute_reply.started":"2024-08-06T02:09:07.928151Z","shell.execute_reply":"2024-08-06T02:09:07.934741Z"},"jupyter":{"outputs_hidden":false}}
original_test['date'].iloc[-1] # 2017-08-31 is test end

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:09:48.881026Z","iopub.execute_input":"2024-08-06T02:09:48.881431Z","iopub.status.idle":"2024-08-06T02:10:10.231033Z","shell.execute_reply.started":"2024-08-06T02:09:48.881397Z","shell.execute_reply":"2024-08-06T02:10:10.229862Z"},"jupyter":{"outputs_hidden":false}}
original_train['date'] = pd.to_datetime(original_train['date'])
original_train['year'] = original_train['date'].dt.year
original_train['month'] = original_train['date'].dt.month
 
monthly_sales = original_train.groupby(['family', 'year','month']).agg({"sales" : "sum"}).reset_index()
 
# The value of the last month (for each 33 products) we change to nan, as otherwise it will distort
# the graph since this month's data is incomplete:
for x in range(33):
    z = 55+(x*56)
    monthly_sales.at[z,'sales'] = np.nan 
 
# We use seaborn's FacetGrid with a col_wrap of 3 to show all the graphs in rows of three.
# We also need sharey = False so that the y axis of all the graphs is not shared but individual.
product_lineplots = sns.FacetGrid(monthly_sales, col="family", hue='year', sharey=False, height=3.5, col_wrap=3, palette='rocket_r')
product_lineplots.map(sns.lineplot, "month", 'sales')
product_lineplots.add_legend()
product_lineplots.set(xlim=(1, 12), ylim=(0, None), xticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:17:50.944951Z","iopub.execute_input":"2024-08-06T02:17:50.945382Z","iopub.status.idle":"2024-08-06T02:17:51.561799Z","shell.execute_reply.started":"2024-08-06T02:17:50.945340Z","shell.execute_reply":"2024-08-06T02:17:51.560604Z"},"jupyter":{"outputs_hidden":false}}
# Create a graph for allsales:
 
total_monthly_sales = original_train.groupby(['year','month']).agg({"sales" : "sum"}).reset_index()
 
total_monthly_sales.at[55,'sales'] = np.nan
 
total_plot = sns.lineplot(x='month', y='sales', hue='year', palette='rocket_r', data=total_monthly_sales)
total_plot.set(xlim=(1, 12), ylim=(0, None), xticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:32:10.084805Z","iopub.execute_input":"2024-08-06T02:32:10.085224Z","iopub.status.idle":"2024-08-06T02:32:13.351746Z","shell.execute_reply.started":"2024-08-06T02:32:10.085191Z","shell.execute_reply":"2024-08-06T02:32:13.350598Z"},"jupyter":{"outputs_hidden":false}}
def create_date_df(df, store_nr):
 
    single_store_df = df[df['store_nbr'] == store_nr]
    single_store_series = single_store_df.groupby(["date"]).sum(numeric_only=True)
 
    return single_store_series
def create_payday_anchors(df):
 
    df.reset_index(inplace=True)
    df['Payday'] = 0
 
    for id, row in df.iterrows():
 
        if search('-01$', row['date']):
            df.at[id - 1, 'Payday'] = 1
 
        if search('-15$', row['date']):
            df.at[id, 'Payday'] = 1
 
    df = df[:-1]
 
    return df
def onehotencode(df, list_of_variables):
 
    column_name_list = list()
    my_category_list = list()
 
    for column in list_of_variables:
 
        categories = df[column].unique().tolist()
 
        for i in categories:
 
            this_list = ((df[column] == i) * 1).tolist()
 
            column_name_list.append(column + str(i))
            my_category_list.append(this_list)
 
            print('Finished ' + str(i))
 
        print(str(column) + ' is done.')
 
    onehotencode_df = pd.DataFrame(my_category_list).transpose()
    onehotencode_df.columns = np.asarray(column_name_list)
 
    return onehotencode_df
def independant_pipeline():
 
    original_train = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv')
    original_test = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv')
 
    # Get one store one product DF:
    one_store_df = create_date_df(original_train, 1)
    one_store_df_test = create_date_df(original_test, 1)
    one_store_df.drop('sales', axis=1, inplace=True)
 
    one_store_df = pd.concat([one_store_df, one_store_df_test])
 
    del original_train
    del original_test
 
    ########################
    # Add Paydays          #
    ########################
 
    one_store_df = create_payday_anchors(one_store_df)
 
    payday_series = one_store_df['Payday']
    payday_count = 0
    payday_scale_list = list()
 
    for x in range(payday_series.__len__()):
 
        if payday_series[x] == 1:
            payday_count = 0
            payday_scale_list.append(payday_count)
        else:
            payday_count += 1
            payday_scale_list.append(payday_count)
 
    one_store_df['Payday_Scale'] = payday_scale_list
 
    one_store_df.drop(['id'], axis=1, inplace=True)
 
    ######################
    # Add Date Variables #
    ######################
 
    dayoftheweek_list = list()
    dayoftheyear_list = list()
    monthoftheyear_list = list()
    year_list = list()
 
    for x in range(1700): # because 1700 different days
 
        thisdate = one_store_df['date'][x]
        thisdayoftheweek = dt.datetime.strptime(thisdate, '%Y-%m-%d').strftime('%A')
        thisdayoftheyear = dt.datetime.strptime(thisdate, '%Y-%m-%d').strftime('%j')
        thismonthoftheyear = dt.datetime.strptime(thisdate, '%Y-%m-%d').strftime('%B')
        thisyear = dt.datetime.strptime(thisdate, '%Y-%m-%d').strftime('%Y')
 
        dayoftheweek_list.append(thisdayoftheweek)
        dayoftheyear_list.append(thisdayoftheyear)
        monthoftheyear_list.append(thismonthoftheyear)
        year_list.append(thisyear)
 
    one_store_df['DayOfTheWeek'] = dayoftheweek_list
    one_store_df['DayOfTheYear'] = dayoftheyear_list
    one_store_df['MonthOfTheYear'] = monthoftheyear_list
    one_store_df['Year'] = year_list
 
    one_store_df['DayOfTheYear'] = pd.to_numeric(one_store_df['DayOfTheYear'])
 
    # Convert DayOfTheWeek to numeric:
 
    dayoftheweek_scale_dict = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
                               'Friday': 5, 'Saturday': 6, 'Sunday': 7}
 
    one_store_df['dayoftheweek_scale'] = one_store_df['DayOfTheWeek'].map(dayoftheweek_scale_dict)
 
    ########################
    # OneHotEncode         #
    ########################
 
    onehotcolumnlist = ('DayOfTheWeek', 'MonthOfTheYear', 'Year')
    onehotencode_df = onehotencode(one_store_df, onehotcolumnlist)
    one_store_df = pd.concat([one_store_df, onehotencode_df], axis=1)
 
    ########################
    # Drop Some Cols       #
    ########################
 
    one_store_df.drop(['store_nbr', 'DayOfTheWeek', 'onpromotion'], axis=1, inplace=True)
 
    return one_store_df
# --- Execute Pipeline --- #
 
independant_df = independant_pipeline()

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:32:26.719166Z","iopub.execute_input":"2024-08-06T02:32:26.719632Z","iopub.status.idle":"2024-08-06T02:32:38.632236Z","shell.execute_reply.started":"2024-08-06T02:32:26.719591Z","shell.execute_reply":"2024-08-06T02:32:38.631018Z"},"jupyter":{"outputs_hidden":false}}
def create_multi_store_one_product_df(df, product_name):
 
    multistore_single_product = df[df['family'] == product_name]
 
    return multistore_single_product
def create_holiday_variables(df):
 
    holidays = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/holidays_events.csv')
 
    holidays = holidays[holidays['transferred'] == False]
    holidays['holiday_type'] = holidays['type']
    holidays.drop(['transferred', 'description', 'type'], axis=1, inplace=True)
 
    national_holidays = holidays[holidays['locale'] == 'National']
    national_holidays['national_holiday_type'] = national_holidays['holiday_type']
    national_holidays.drop(['locale', 'locale_name', 'holiday_type'], axis=1, inplace=True)
    national_holidays.drop_duplicates(subset='date', keep="first", inplace=True)
    df = pd.merge(df, national_holidays, how='left', on=['date'])
 
    state_holidays = holidays[holidays['locale'] == 'Regional']
    state_holidays['state'] = state_holidays['locale_name']
    state_holidays['state_holiday_type'] = state_holidays['holiday_type']
    state_holidays.drop(['locale', 'locale_name', 'holiday_type'], axis=1, inplace=True)
    df = pd.merge(df, state_holidays, how='left', on=['date', 'state'])
 
    city_holidays = holidays[holidays['locale'] == 'Local']
    city_holidays['city'] = city_holidays['locale_name']
    city_holidays['city_holiday_type'] = city_holidays['holiday_type']
    city_holidays.drop(['locale', 'locale_name', 'holiday_type'], axis=1, inplace=True)
    city_holidays.drop([265], axis=0, inplace=True)
    df = pd.merge(df, city_holidays, how='left', on=['date', 'city'])
 
    df['holiday_type'] = np.nan
    df['holiday_type'] = df['holiday_type'].fillna(df['national_holiday_type'])
    df['holiday_type'] = df['holiday_type'].fillna(df['state_holiday_type'])
    df['holiday_type'] = df['holiday_type'].fillna(df['city_holiday_type'])
    df.drop(['national_holiday_type', 'state_holiday_type', 'city_holiday_type'], axis=1, inplace=True)
 
    return df
def create_location_variables(df):
 
    stores = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv')
 
    # stores['city'].unique()
    # ['Quito', 'Santo Domingo', 'Cayambe', 'Latacunga', 'Riobamba',
    #  'Ibarra', 'Guaranda', 'Puyo', 'Ambato', 'Guayaquil', 'Salinas',
    #  'Daule', 'Babahoyo', 'Quevedo', 'Playas', 'Libertad', 'Cuenca',
    #  'Loja', 'Machala', 'Esmeraldas', 'Manta', 'El Carmen']
 
    # Height dict:
    Height = {'Quito': 2850, 'Santo Domingo': 550, 'Cayambe': 2830, 'Latacunga': 2860,
              'Riobamba': 2754, 'Ibarra': 2225, 'Guaranda': 2668, 'Puyo': 950,
              'Ambato': 2577, 'Guayaquil': 0, 'Salinas': 0, 'Daule': 0,
              'Babahoyo': 0, 'Quevedo': 75, 'Playas': 0, 'Libertad': 36,
              'Cuenca': 2560, 'Loja': 2060, 'Machala': 0, 'Esmeraldas': 15,
              'Manta': 0, 'El Carmen': 250}
 
    # Elevation:
    # 0 = 0 - 200 (10)
    # 1 = 200-700 (2)
    # 2 = 700-1500 (1)
    # 3 = 1500-2300 (2)
    # 4 = 2300-3000 (7)
 
    Population = {'Quito': 2000000, 'Santo Domingo': 460000, 'Cayambe': 40000, 'Latacunga': 100000,
                  'Riobamba': 157000, 'Ibarra': 150000, 'Guaranda': 35000, 'Puyo': 40000,
                  'Ambato': 350000, 'Guayaquil': 2750000, 'Salinas': 50000, 'Daule': 130000,
                  'Babahoyo': 105000, 'Quevedo': 200000, 'Playas': 40000, 'Libertad': 105000,
                  'Cuenca': 445000, 'Loja': 200000, 'Machala': 260000, 'Esmeraldas': 200000,
                  'Manta': 240000, 'El Carmen': 120000}
 
    # Population:
    # 0 = 0-60000 (5)
    # 1 = 60000-160000 (12)
    # 2 = 160000-280000 (3)
    # 3 = 280000+ (2)
 
    Size = {'Quito': 372, 'Santo Domingo': 60, 'Cayambe': 378, 'Latacunga': 370,
            'Riobamba': 59, 'Ibarra': 242, 'Guaranda': 520, 'Puyo': 88,
            'Ambato': 47, 'Guayaquil': 345, 'Salinas': 27, 'Daule': 475,
            'Babahoyo': 175, 'Quevedo': 300, 'Playas': 280, 'Libertad': 28,
            'Cuenca': 71, 'Loja': 44, 'Machala': 67, 'Esmeraldas': 70,
            'Manta': 60, 'El Carmen': 1250}
 
    stores["City_Population"] = stores['city'].map(Population)
    stores["City_Elevation"] = stores['city'].map(Height)
    stores["City_Size"] = stores['city'].map(Size)
    stores["City_Density"] = round(stores["City_Population"] / stores["City_Size"],0)
    stores["City_Population_Category"] = 0
    stores["City_Elevation_Category"] = 0
    stores["City_Size_Category"] = 0
    stores["City_Density_Category"] = 0
 
    for id, row in stores.iterrows():
 
        if row['City_Elevation'] < 200:
            stores.at[id, 'City_Elevation_Category'] = 0
        elif row['City_Elevation'] < 700:
            stores.at[id, 'City_Elevation_Category'] = 1
        elif row['City_Elevation'] < 1500:
            stores.at[id, 'City_Elevation_Category'] = 2
        elif row['City_Elevation'] < 2300:
            stores.at[id, 'City_Elevation_Category'] = 3
        else:
            stores.at[id, 'City_Elevation_Category'] = 4
 
        if row['City_Population'] < 60000:
            stores.at[id, 'City_Population_Category'] = 0
        elif row['City_Population'] < 160000:
            stores.at[id, 'City_Population_Category'] = 1
        elif row['City_Population'] < 280000:
            stores.at[id, 'City_Population_Category'] = 2
        else:
            stores.at[id, 'City_Population_Category'] = 3
 
        if row['City_Size'] < 150:
            stores.at[id, 'City_Size_Category'] = 0
        elif row['City_Size'] < 325:
            stores.at[id, 'City_Size_Category'] = 1
        elif row['City_Size'] < 1000:
            stores.at[id, 'City_Size_Category'] = 2
        else:
            stores.at[id, 'City_Size_Category'] = 3
 
        if row['City_Density'] < 150:
            stores.at[id, 'City_Density_Category'] = 0
        elif row['City_Density'] < 325:
            stores.at[id, 'City_Density_Category'] = 1
        elif row['City_Density'] < 1000:
            stores.at[id, 'City_Density_Category'] = 2
        elif row['City_Density'] < 3000:
            stores.at[id, 'City_Density_Category'] = 3
        elif row['City_Density'] < 7000:
            stores.at[id, 'City_Density_Category'] = 4
        else:
            stores.at[id, 'City_Density_Category'] = 5
 
    city_variables_df = stores[['store_nbr', 'City_Elevation_Category', 'City_Population_Category', 'City_Size_Category',
                                'City_Density_Category', 'City_Density']]
    df = pd.merge(df, city_variables_df, how='left', on='store_nbr')
    df.drop(['city','state'], axis=1, inplace=True)
 
    return df
def all_stores_pipeline():
 
    originaltrainFull = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv')
    originaltest = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv')
 
    stores = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/stores.csv')
    transactions = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/transactions.csv')
 
    all_stores_df = create_multi_store_one_product_df(originaltrainFull, 'AUTOMOTIVE')
    all_stores_df_test = create_multi_store_one_product_df(originaltest, 'AUTOMOTIVE')
    all_stores_df.drop('sales', axis=1, inplace=True)
 
    all_stores_df = pd.concat([all_stores_df, all_stores_df_test])
    all_stores_df.drop(['id', 'family', 'onpromotion'], axis=1, inplace=True)
 
    all_stores_df = pd.merge(all_stores_df, stores, how='left', on=['store_nbr'])
 
    del originaltest
    del originaltrainFull
 
    #########################
    # Add Holiday Variables #
    #########################
 
    all_stores_df = create_holiday_variables(all_stores_df)
 
    ##########################
    # Add Location Variables #
    ##########################
 
    all_stores_df = create_location_variables(all_stores_df)
 
    ################################
    # Create Store Closed Variable #
    ################################
 
    all_stores_df = pd.merge(all_stores_df, transactions, how='left', on=['date', 'store_nbr'])
    all_stores_df['transactions'].fillna(0, inplace=True)
 
    store_closed = [1 if x == 0 else 0 for x in all_stores_df['transactions']]
 
    all_stores_df['store_closed'] = store_closed
    all_stores_df['store_closed'].iloc[-864:] = 0
 
    all_stores_df.drop('transactions', axis=1, inplace=True)
 
    ###################
    # OneHotEncode    #
    ###################
 
    all_stores_df['isholiday'] = 1
    thislist = all_stores_df['holiday_type'].isna()
    all_stores_df.loc[thislist,'isholiday'] = 0
 
    onehotcolumnlist = ('store_nbr', 'type', 'cluster', 'holiday_type', 'City_Elevation_Category',
                        'City_Population_Category', 'City_Density_Category', 'City_Size_Category')
 
    onehotencode_df = onehotencode(all_stores_df, onehotcolumnlist)
    all_stores_df = pd.concat([all_stores_df, onehotencode_df], axis=1)
 
    ###################
    # Drop Some Cols  #
    ###################
 
    all_stores_df.drop(['type', 'cluster', 'holiday_type'], axis=1, inplace=True)
 
    return all_stores_df
# --- Execute Pipeline --- #
 
all_stores_df = all_stores_pipeline()

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T02:58:23.150884Z","iopub.execute_input":"2024-08-06T02:58:23.151923Z","iopub.status.idle":"2024-08-06T03:03:25.931937Z","shell.execute_reply.started":"2024-08-06T02:58:23.151885Z","shell.execute_reply":"2024-08-06T03:03:25.930791Z"},"jupyter":{"outputs_hidden":false}}
def full_product_pipeline(family, independant_df=independant_df, all_stores_df=all_stores_df):
 
    originaltrainFull = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/train.csv')
    originaltest = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/test.csv')
 
    multistore_product = create_multi_store_one_product_df(originaltrainFull, family)
 
    # merge with test:
    multistore_product_test = create_multi_store_one_product_df(originaltest, family)
    multistore_product_test['sales'] = np.nan
 
    del originaltrainFull
    del originaltest
 
    # take log of sales:
    multistore_product['sales'] = np.log1p(multistore_product['sales']+1)
 
    msp_full = pd.concat([multistore_product, multistore_product_test])
 
    # reset index:
    msp_full.reset_index(inplace=True, drop=True)
 
    ######################
    # Add Independant DF #
    ######################
 
    msp_full = pd.merge(msp_full, independant_df, how='left', on=['date'])
 
    #####################
    # Add All Stores DF #
    #####################
 
    msp_full = pd.merge(msp_full, all_stores_df, how='left', on=['date', 'store_nbr'])
 
    ############################
    # Add Earthquake Info      #
    ############################
 
    earthquake_day = [1 if x == '2016-04-16' else 0 for x in msp_full['date']]
    earthquake_impact = [1 if (x > '2016-04-16') & (x < '2016-05-16') else 0 for x in msp_full['date']]
 
    msp_full['earthquake_day'] = earthquake_day
    msp_full['earthquake_impact'] = earthquake_impact
 
    ############################
    # Add School Info          #
    ############################
 
    school_preparation = [1 if (x > '2014-09-15') & (x < '2014-10-15') or (x > '2015-09-15') & (x < '2015-10-15')
                          or (x > '2016-09-15') & (x < '2016-10-15') or (x > '2017-09-15') & (x < '2017-10-15')
                          else 0 for x in msp_full['date']]
 
    msp_full['school_preparation'] = school_preparation
 
    #############################
    # Clean DF before modelling #
    #############################
 
    msp_full.drop(['family', 'MonthOfTheYear'], axis=1, inplace=True)
 
    return msp_full
# --- Execute Full Product Pipeline for each product --- #
 
# List all product families:
 
list_of_families = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS',
                    'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS',
                    'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
                    'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES',
                    'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE',
                    'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE',
                    'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY',
                    'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES',
                    'SEAFOOD']
 
# Create new .csv for each product family:
 
for x in list_of_families:
 
    this_df = full_product_pipeline(x)
 
    if x == 'BREAD/BAKERY':
 
            x = 'BREADBAKERY'
 
    print('Completed eda for ' + str(x))
    this_df.to_csv('/kaggle/working/'+str(x)+'.csv', index=False)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T03:03:25.933731Z","iopub.execute_input":"2024-08-06T03:03:25.934096Z","iopub.status.idle":"2024-08-06T03:03:27.406670Z","shell.execute_reply.started":"2024-08-06T03:03:25.934064Z","shell.execute_reply":"2024-08-06T03:03:27.405378Z"},"jupyter":{"outputs_hidden":false}}
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMRegressor
 
sample_submission = pd.read_csv('/kaggle/input/store-sales-time-series-forecasting/sample_submission.csv')
def scorethis_rmsle(prediction_list, y_list):
 
    scorelist = list()
 
    for x in range(prediction_list.__len__()):
 
 
        log_score_x = np.abs(np.abs(prediction_list[x]) - np.abs(y_list[x]))
        
        try:
            [scorelist.append(y) for y in log_score_x.values]
        except:
            scorelist.append(log_score_x)
 
    score_array = np.array(scorelist)
 
    rmsle = np.sqrt(np.mean(score_array**2)) # sqrt of mean of power of difference of the logs
    rmsle = np.round(rmsle, 3)
 
    return rmsle
def create_validation(this_family_df, validation=True):
    
    if validation is True:
    
        this_family_df = this_family_df[:-864]
        # Remove the 864 top submission rows if it is for validation
    
    this_family_sales = this_family_df['sales']
 
    this_family_df.drop(['sales', 'date'], axis=1, inplace=True)
 
    ########################
    # Scale Data           #
    ########################
 
    scaler = MinMaxScaler()
    this_family_df[this_family_df.columns] = scaler.fit_transform(this_family_df[this_family_df.columns])
 
    ########################
    # Split Train and Test #
    ########################
 
    test = this_family_df.iloc[-864:]
    test_y = this_family_sales.iloc[-864:]
 
    train = this_family_df.iloc[:-864]
    train_y = this_family_sales.iloc[:-864]
 
    return train, train_y, test, test_y
def lgbmr_run(train, train_y, test, test_y,
           validation=True):
    
    #################
    # Create Model  #
    #################
 
    lgbmr_model = LGBMRegressor(
        colsample_bytree=0.7,
        learning_rate=0.055,
        min_child_samples=10,
        num_leaves=19,
        objective='regression',
        n_estimators=1000,
        n_jobs=4,
        random_state=337)
 
    #################
    # Execute LGBMR #
    #################
 
    lgbmr_model.fit(train, train_y)
    lgbmr_pred = lgbmr_model.predict(test).tolist()
    lgbmr_pred = [round(x, 2) for x in lgbmr_pred]
    
    if validation == True:
        
        # validation set also has ground truths:
        test_y = test_y.to_list()
 
        return lgbmr_pred, test_y
 
    else:
 
        return lgbmr_pred
def execute_validation(thisfunc):
 
    double_list_of_predictions = []
    double_list_of_ground_truths = []
 
    for x in list_of_families: # 33
        
        if x == 'BREAD/BAKERY':
 
            x = 'BREADBAKERY'
            # Otherwise would create an error searching for the BREAD/ directory instead of the file
 
        print('Evaluating '+str(x)+'...')
        
        this_df = pd.read_csv('/kaggle/working/' + str(x) + '.csv')
 
        train, train_y, test, test_y = create_validation(this_df)
        pred, y = thisfunc(train, train_y, test, test_y, validation=True)
        
        if x == 'BOOKS':
 
            zero_list = []
 
            for g in range(864):
 
                zero_list.append(0.6931471805599453) 
                # this will be exactly 0 when we transform our predictions again
                # to before we did log(sales +1)
 
            double_list_of_predictions.append(zero_list)
            double_list_of_ground_truths.append(y) 
            
        else:
            
            double_list_of_predictions.append(pred) # 33 * [864]
            double_list_of_ground_truths.append(y) # 33 * [864]
 
    list_of_predictions = list()
    list_of_ground_truths = list()
 
    for x in double_list_of_predictions:
        for y in x:
            list_of_predictions.append(y) # unpack 33 * 864
 
    for x in double_list_of_ground_truths:
        for z in x:
            list_of_ground_truths.append(z) # unpack 33 * 864
 
    return list_of_predictions, list_of_ground_truths
# --- Execute LGBMR Model On Validation Set --- #
 
# Run this code if you want to do a validation test + see the score:
 
# list_of_lgbmr_predictions, list_of_ground_truths = execute_validation(lgbmr_run)
# scorethis_rmsle(list_of_lgbmr_predictions, list_of_ground_truths)

# %% [code] {"execution":{"iopub.status.busy":"2024-08-06T03:22:27.729116Z","iopub.execute_input":"2024-08-06T03:22:27.730268Z","iopub.status.idle":"2024-08-06T03:28:52.637118Z","shell.execute_reply.started":"2024-08-06T03:22:27.730227Z","shell.execute_reply":"2024-08-06T03:28:52.635984Z"},"jupyter":{"outputs_hidden":false}}
def execute_submission(thisfunc):
 
    list_of_predictions = []
 
    for x in list_of_families:
        
        if x == 'BREAD/BAKERY':
 
            x = 'BREADBAKERY'
            # Otherwise would create an error searching for the BREAD/ directory instead of the file
 
        print('Evaluating '+str(x)+'...')
        this_df = pd.read_csv('/kaggle/working/' + str(x) + '.csv')
        
        if x == 'BOOKS':
 
            zero_list = []
 
            for g in range(864):
 
                zero_list.append(0.6931471805599453) 
                # this will be exactly 0 when we transform our predictions again
                # to before we did log(sales +1)
 
            list_of_predictions.append(zero_list)
 
        else:
    
            train, train_y, test, test_y = create_validation(this_df, validation=False)
            pred = thisfunc(train, train_y, test, test_y=None, validation=False)
            list_of_predictions.append(pred)
    
    ###############################
    # Put Back In Submission Form # 
    ###############################
    
    restructured_predictions = list()
 
    for y in range(864):
 
        for z in range(33):
            restructured_predictions.append(list_of_predictions[z][y])
 
    restructured_predictions = np.expm1(restructured_predictions) - 1
 
    return restructured_predictions
# --- Execute Submission --- #
 
restructured_predictions = execute_submission(lgbmr_run)
sample_submission['sales'] = restructured_predictions
 
# Convert some (slightly) negative predictions to a zero prediction:
sample_submission['sales'] = [0 if x < 0 else x for x in sample_submission['sales']]
 
sample_submission.to_csv('/kaggle/working/submission.csv', index=False)

# %% [code] {"jupyter":{"outputs_hidden":false}}
