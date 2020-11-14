import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from tqdm import tqdm
import datetime
from threading import Thread
from queue import Queue

import os, glob

DATA_PATH = './data/'

month_files = ['2019-Oct.csv', '2019-Nov.csv']
total_records = 109950743 
dtype = {'price':np.float32, 'product_id':np.int32, 'user_id':np.int32}

########## UTILITY ROUTINES ############################################################################################################

def get_needed_data(month_files:list, columns:list, parse_dates=False, concat=True):
    global DATA_PATH, dtype
    dataframes = pd.DataFrame(columns=columns) if concat else []
    date_parser = pd.to_datetime if parse_dates else None
    infer_datetime = True if parse_dates else False
    if concat:
        for month in month_files:
            dataframes = pd.concat([dataframes, pd.read_csv(os.path.join(DATA_PATH, month), usecols=columns, parse_dates=parse_dates, date_parser=date_parser, infer_datetime_format=infer_datetime, dtype=dtype)], ignore_index=True)
    else:
        for month in month_files:
            dataframes.append(pd.read_csv(os.path.join(DATA_PATH, month), usecols=columns, parse_dates=parse_dates, date_parser=date_parser, infer_datetime_format=infer_datetime))
    return dataframes


def get_needed_iterator(month_files:list, columns:list, chunksize:int=10**7, parse_dates=False):
    global DATA_PATH, dtype
    iterators = []
    date_parser = pd.to_datetime if parse_dates else None
    for month in month_files:
        iterators.append(pd.read_csv(os.path.join(DATA_PATH, month), usecols=columns, parse_dates=parse_dates, date_parser=date_parser, chunksize=chunksize, dtype=dtype))
    return iterators


def load_data(month_files:list, columns:list, chunk=False, chunksize:int=10**6, parse_dates=False):
    if chunk:
        return get_needed_iterator(month_files, columns, chunksize, parse_dates)
    else:
        return get_needed_data(month_files, columns, parse_dates)


def compute_by_chunk(chunks, function, **kargs):
    results = []
    for df in tqdm(chunks):
        results.append(function(df, **kargs))
    return results


def clean_memory(garbage:list):
    size = len(garbage)
    for i in range(size-1, -1, -1):
        del garbage[i]


def get_purchases(columns:list):
    global month_files
    purchases_months_list = []
    for month in month_files:
        iterator = load_data([month, ], columns, chunk=True)[0]
        purchases_list = []
        for dataset in iterator:
            purchases_list.append(dataset[dataset.event_type == 'purchase'])
        purchases_months_list.append(pd.concat(purchases_list, ignore_index=True))
        clean_memory(purchases_list)
    return purchases_months_list




########## RQ1 ############################################################################################################

## ***** Which is the rate of complete funnels? (WORKS)
def complete_funnels(columns:list):
    global month_files, total_records
    purchases_list = []
    carts_list = []
    dataset_iterators = load_data(month_files, columns, chunk=True)
    for iterator in dataset_iterators:
        for dataset in iterator:
            purchases_list.append(dataset[dataset.event_type == 'purchase'])
            carts_list.append(dataset[dataset.event_type == 'cart'])
    purchases = pd.concat(purchases_list, ignore_index=True)
    clean_memory(purchases_list)
    carts = pd.concat(carts_list, ignore_index=True)
    clean_memory(carts_list)
    n_purchases = purchases.size
    n_carts = carts.size
    purchases_carts = purchases.merge(carts, on=['user_session', 'product_id'], how='left', indicator=True)
    clean_memory([purchases, carts])
    n_one_click = purchases_carts[purchases_carts._merge == 'left_only'].size
    clean_memory([purchases_carts, ])
    print(f'There are {n_purchases} purchases,\nof which {n_one_click} one_click purchases (only view-purchase couples).\nThe number of carts is {n_carts}.', end='\n'*3)
    # completed = (n_purchases - n_one_click) / (total_records - 3*(n_purchases - n_one_click) - 2*(n_carts + n_one_click - n_purchases) - 2*n_one_click) * 100
    completed = [0.0] * 2
    completed[0] = (n_purchases - n_one_click) / (total_records - n_purchases - n_carts) * 100 # completed without one_click purchases
    completed[1] = n_purchases / (total_records - n_purchases - n_carts) * 100 # completed with one_click purchases
    print(f'Rate of complete funnels without one_click purchases:\t{round(completed[0], 2)}%.\n\nRate of complete funnels with one_click purchases:\t{round(completed[1], 2)}%.')



## ***** What’s the operation users repeat more on average within a session? (WORKS)
def average_number_operations(columns:list):
    global month_files
    dataset_iterators = load_data(month_files, columns, chunk=True)
    sessions_list = []
    n_views, n_carts, n_purchases = (0,0,0)
    for iterator in dataset_iterators:
        for dataset in iterator:
            n_views += dataset[dataset.event_type == 'view'].size
            n_carts += dataset[dataset.event_type == 'cart'].size
            n_purchases += dataset[dataset.event_type == 'purchase'].size
            sessions_list.append(dataset['user_session'].drop_duplicates())
            clean_memory([dataset, ])
    n_sessions = pd.concat(sessions_list, ignore_index=True).drop_duplicates().size
    print(f'{round(n_views / n_sessions)} views, {round(n_carts / n_sessions)} carts, {round(n_purchases / n_sessions)} purchases. {n_sessions} sessions.\n\n')
    plt.figure(figsize=(8,8))
    plt.bar(['views', 'carts', 'purchases'], [round(n_views / n_sessions, 2), round(n_carts / n_sessions, 2), round(n_purchases / n_sessions, 2)])
    plt.title('Average number for each operation within a session')
    plt.xlabel('Operation')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()



## ***** How many times, on average, a user views a product before adding it to the cart? (WORKS)
def average_views_before_cart(columns:list):
    global month_files
    carts_list = []
    views_carts_list = []
    dataset_iterators = load_data(month_files, columns, chunk=True, parse_dates=['event_time'])
    for iterator in dataset_iterators:
        for dataset in iterator:
            carts_list.append(dataset[dataset.event_type == 'cart'])
    carts = pd.concat(carts_list, ignore_index=True)
    clean_memory(carts_list)
    dataset_iterators = load_data(month_files, columns, chunk=True, parse_dates=['event_time'])
    for iterator in dataset_iterators:
        for dataset in iterator:
            views_per_chunk = dataset[dataset.event_type == 'view']
            views_carts_per_chunk = views_per_chunk.merge(carts, on=['user_id', 'product_id'], suffixes=['_view', '_cart'])
            del views_per_chunk
            views_carts_per_chunk = views_carts_per_chunk[views_carts_per_chunk.event_time_view < views_carts_per_chunk.event_time_cart]
            views_carts_list.append(views_carts_per_chunk)
            del views_carts_per_chunk
    views_carts = pd.concat(views_carts_list, ignore_index=True)
    clean_memory(views_carts_list)
    avg = views_carts.groupby(['user_id', 'product_id']).event_time_view.count().mean()
    clean_memory([views_carts,])
    print(f'On average a user views a product {round(avg)} times before adding it to the cart.')



## ***** What’s the probability that products added once to the cart are effectively bought? (WORKS)
def probability_purchase(columns:list):
    global month_files
    carts_list = []
    purchases_list = []
    dataset_iterators = load_data(month_files, columns, chunk=True)
    for iterator in dataset_iterators:
        for dataset in iterator:
            carts_list.append(dataset[dataset.event_type == 'cart'])
            purchases_list.append(dataset[dataset.event_type == 'purchase'])
    carts = pd.concat(carts_list, ignore_index=True)
    clean_memory(carts_list)
    purchases = pd.concat(purchases_list, ignore_index=True)
    clean_memory(purchases_list)
    carts_purchases = carts.merge(purchases.drop_duplicates(), on=['user_session', 'product_id'], how='left', indicator=True)
    n_carts = carts.size
    clean_memory([carts, purchases])
    n_purchased = carts_purchases[carts_purchases._merge == 'both'].size
    print(f'The probability that products added once to the cart are effectively bought is {round(n_purchased / n_carts * 100, 2)}%.') # carts_purchases.size



## ***** What’s the average time an item stays in the cart before being removed?
def average_time_cart(columns:list):
    global month_files
    last_session_op_list = []
    carts_list = []
    purchases_list = []
    dataset_iterators = load_data(month_files, columns, chunk=True, parse_dates=['event_time'])
    for iterator in dataset_iterators:
        for dataset in tqdm(iterator):
            carts_list.append(dataset[dataset.event_type == 'cart'])
            purchases_list.append(dataset[dataset.event_type == 'purchase'])
            last_session_op_list.append(dataset.groupby('user_session').event_time.max())
            clean_memory([dataset, ])
    carts = pd.concat(carts_list, ignore_index=True)
    purchases = pd.concat(purchases_list, ignore_index=True)
    clean_memory([carts_list, purchases_list])
    pending_carts = carts.merge(purchases, on=['user_session', 'product_id'], how='left', indicator=True)
    pending_carts = pending_carts[pending_carts._merge == 'left_only']
    clean_memory([carts, purchases])
    last_session_op = pd.concat(last_session_op_list).groupby('user_session').max()
    time_deltas = 0
    for session, max_time in last_session_op.items():
        # time_deltas += pending_carts[pending_carts.user_session == session].event_time_x.apply(lambda x: (max_time - x).total_seconds()).sum()
        for time in pending_carts[pending_carts.user_session == session]['event_time_x']:
            time_deltas += (max_time - time).total_seconds()
    print(f'The average time an item stays in the cart before being removed is {round(time_deltas / pending_carts.size, 2):,.2f} seconds.')



## ***** How much time passes on average between the first view time and a purchase/addition to cart?
def average_time_after_first_view_1(columns:list):
    global month_files
    user_views_p_min_list = []
    user_purchases_carts_p_min_list = []
    dataset_iterators = load_data(month_files, columns, chunk=True, parse_dates=['event_time'])
    for iterator in dataset_iterators:
        for dataset in iterator:
            user_views_p_min_list.append(dataset[dataset.event_type == 'view'][['user_id', 'event_time']].groupby('user_id').event_time.min())
            user_purchases_carts_p_min_list.append(dataset[(dataset.event_type == 'cart') | (dataset.event_type == 'purchase')][['user_id', 'event_time']].groupby('user_id').event_time.min())
            clean_memory([dataset, ])
    user_views_min = pd.concat(user_views_p_min_list).groupby('user_id').min()
    clean_memory(user_views_p_min_list)
    user_purchases_carts_min = pd.concat(user_purchases_carts_p_min_list).groupby('user_id').min()
    clean_memory(user_purchases_carts_p_min_list)
    joined = user_views_min.to_frame().merge(user_purchases_carts_min.to_frame(), on='user_id', suffixes=['_views', '_purchases_carts'])
    sum_deltas = 0
    for t_views, t_purchases_carts in zip(joined['event_time_views'], joined['event_time_purchases_carts']):
        sum_deltas += (t_purchases_carts - t_views).total_seconds()
    mean = round(sum_deltas / joined.size, 2)
    st_dev = 0
    for t_views, t_purchases_carts in zip(joined['event_time_views'], joined['event_time_purchases_carts']):
        st_dev += ((t_purchases_carts - t_views).total_seconds() - mean)**2
    st_dev = round(np.sqrt(st_dev / (joined.size - 1)), 2) 
    print(f'The average time between the first view time and a purchase/addition to cart is {round(mean / 3600, 2)} hours.\nThe standard deviation is {round(st_dev / 3600, 2)} hours.') # we convert the time from seconds to hours



def average_time_after_first_view_2(columns:list):
    global month_files
    user_views_p_min_list = []
    user_purchases_carts_p_min_list = []
    dataset_iterators = load_data(month_files, columns, chunk=True, chunksize=2*10**7, parse_dates=['event_time'])
    with open('../data/views_p_min.csv', 'a') as v_f, open('../data/cart_purchases_p_min.csv', 'a') as cp_f:
        for iterator in dataset_iterators:
            for dataset in tqdm(iterator):
                user_views_p_min_list.append(dataset[dataset.event_type == 'view'][['user_id', 'product_id', 'event_time']].groupby(['user_id', 'product_id']).event_time.min())
                user_purchases_carts_p_min_list.append(dataset[(dataset.event_type == 'cart') | (dataset.event_type == 'purchase')][['user_id', 'product_id', 'event_time']].groupby(['user_id', 'product_id']).event_time.min())
                del dataset
    user_views_min = pd.concat(user_views_p_min_list).groupby(['user_id', 'product_id']).min()
    del user_views_p_min_list
    user_purchases_carts_min = pd.concat(user_purchases_carts_p_min_list).groupby(['user_id', 'product_id']).min()
    del user_purchases_carts_p_min_list
    joined = user_views_min.to_frame().merge(user_purchases_carts_min.to_frame(), on=['user_id', 'product_id'], suffixes=['_views', '_purchases_carts'])
    sum_deltas = 0
    for t_views, t_purchases_carts in zip(joined['event_time_views'], joined['event_time_purchases_carts']):
        sum_deltas += (t_purchases_carts - t_views).total_seconds()
    mean = round(sum_deltas / joined.size, 2)
    st_dev = 0
    for t_views, t_purchases_carts in zip(joined['event_time_views'], joined['event_time_purchases_carts']):
        st_dev += ((t_purchases_carts - t_views).total_seconds() - mean)**2
    st_dev = round(np.sqrt(st_dev / joined.size), 2) 
    print(f'The average time between the first view time and a purchase/addition to cart is {round(mean / 60, 2)} minutes.\nThe standard deviation is {round(st_dev / 60, 2)} minutes.') # we convert the time from seconds to hours


# def average_time_after_first_view_2(columns:list):
#     global month_files
#     user_views_min = pd.DataFrame()
#     user_purchases_carts_min = pd.DataFrame()
#     dataset_iterators = load_data(month_files, columns, chunk=True, parse_dates=['event_time'])
#     for iterator in dataset_iterators:
#         for dataset in tqdm(iterator):
#             user_views_min = pd.concat([user_views_min, dataset[dataset.event_type == 'view'][['user_id', 'product_id', 'event_time']].groupby(['user_id', 'product_id']).event_time.min()])
#             user_purchases_carts_min = pd.concat([user_purchases_carts_min, dataset[(dataset.event_type == 'cart') | (dataset.event_type == 'purchase')][['user_id', 'product_id', 'event_time']].groupby(['user_id', 'product_id']).event_time.min()])
#             clean_memory([dataset, ])
#     user_views_min = user_views_min.groupby(['user_id', 'product_id']).min()
#     user_purchases_carts_min = user_purchases_carts_min.groupby(['user_id', 'product_id']).min()
#     joined = user_views_min.to_frame().merge(user_purchases_carts_min.to_frame(), on=['user_id', 'product_id'], suffixes=['_views', '_purchases_carts'])
#     sum_deltas = 0
#     for t_views, t_purchases_carts in zip(joined['event_time_views'], joined['event_time_purchases_carts']):
#         sum_deltas += (t_purchases_carts - t_views).total_seconds()
#     mean = round(sum_deltas / joined.size, 2)
#     st_dev = 0
#     for t_views, t_purchases_carts in zip(joined['event_time_views'], joined['event_time_purchases_carts']):
#         st_dev += ((t_purchases_carts - t_views).total_seconds() - mean)**2
#     st_dev = round(np.sqrt(st_dev / (joined.size - 1)), 2) 
#     print(f'The average time between the first view time and a purchase/addition to cart is {round(mean / 60, 2):,} minutes.\nThe standard deviation is {round(st_dev / 60, 2):,} minutes.') # we convert the time from seconds to hours



########## RQ2 ############################################################################################################

## *****Number of sold products per category
def num_sold_per_cat():
    #import the data of October and parse the column event_time to datetime
    data = pd.read_csv("2019-Oct.csv", #nrows=9000000, 
                   parse_dates=['event_time'],
                   date_parser=pd.to_datetime)
    #import the data of November
    data2 = pd.read_csv("2019-Nov.csv", #nrows=9000000, 
                   parse_dates=['event_time'],
                   date_parser=pd.to_datetime)
    #concatenate the two datasets
    fr=[data,data2]
    data_tot=pd.concat(fr,ignore_index=True)
    #plot showing the number of sold products per category
    # obtain a subset of only the purchases, group by category code, and count the number of sold product plotting the biggest 10 values
    rq2_main=data_tot[data_tot.event_type=='purchase'].groupby(data_tot.category_code)\
    .product_id.count().sort_values(ascending=False).head(10)

    rq2_main.plot.barh()

## *****Most visited subcategories

def most_v_cat():
    #import the data of October and parse the column event_time to datetime
    data = pd.read_csv("2019-Oct.csv", #nrows=9000000, 
                   parse_dates=['event_time'],
                   date_parser=pd.to_datetime)

    #import the data of November
    data2 = pd.read_csv("2019-Nov.csv", #nrows=9000000, 
                   parse_dates=['event_time'],
                   date_parser=pd.to_datetime)

    #concatenate the two datasets
    fr=[data,data2]
    data_tot=pd.concat(fr,ignore_index=True)
    #Plot of the most visited subcategories
    # obtain a subset of only the views, group by category code, and count the views of each category plotting the biggest 10 values
    rq2_1=data_tot[data_tot.event_type=='view'].groupby(data_tot.category_code)\
    .category_code.count().nlargest(10)

    rq2_1.plot.barh()

## *****10 most sold products per category    
 def most_sold_per_cat():
        #import the data of October and parse the column event_time to datetime
        data = pd.read_csv("2019-Oct.csv", #nrows=9000000, 
                   parse_dates=['event_time'],
                   date_parser=pd.to_datetime)

        #import the data of November
        data2 = pd.read_csv("2019-Nov.csv", #nrows=9000000, 
                   parse_dates=['event_time'],
                   date_parser=pd.to_datetime)

        #concatenate the two datasets
        fr=[data,data2]
        data_tot=pd.concat(fr,ignore_index=True)
        #The 10 most sold products per category
        # obtain a subset of only the purchases,
        p=data_tot[data_tot.event_type=='purchase']
        #for each category code count the number of sold product showing the biggest 10 values
        for category_code, t in  p.groupby('category_code'):
            print(category_code)
            print(t.groupby('product_id').user_session.count().nlargest(10))
    
    
    
    
    
########## RQ4 ############################################################################################################

## ***** How much does each brand earn per month? (WORKS)
def get_profit_per_month(brand_name, dataset_list:list=None):
    global month_files
    if dataset_list == None:
        dataset_list = get_purchases(['brand', 'product_id', 'price', 'event_type'])
    profit_per_month = {}
    for i in range(len(month_files)):
        profit = 0
        month_name = month_files[i][5:8]
        brand_items = dataset_list[i][dataset_list[i].brand == brand_name]
        if not brand_items.empty:
            profit = brand_items[(brand_items.price.notna()) & (brand_items.price.notnull())].price.agg('sum')
        profit_per_month[month_name] = profit
    return profit_per_month



## ***** Is the average price of products of different brands significantly different? (WORKS)
def price_std_dev(columns:list):
    global month_files
    dataset_iterators = load_data(month_files, columns, chunk=True)
    brand_products_list = []
    for iterator in dataset_iterators:
        for dataset in iterator:
            brand_products_list.append(dataset.drop_duplicates())
    brand_products = pd.concat(brand_products_list, ignore_index=True).drop_duplicates()
    brand_means = brand_products.groupby('brand').price.mean()
    mean = brand_means.mean()
    std = brand_means.std()
    print(f'The mean of the average price of the products for each brand is {round(mean, 2):,.2f}$,\nwith a standard deviation of {round(std, 2):,.2f}$.')
    return brand_means



## ***** Find the top 3 brands that have suffered the biggest losses in earnings between one month and the next (WORKS)
def top_n_two_months_losses(columns:list, month1:str, month2:str, n=3):
    dataset_list = get_purchases(columns)
    brands = pd.concat(dataset_list, ignore_index=True)
    brands = brands[brands.brand.notna()].brand.unique()
    diff_profit = dict.fromkeys(brands)
    for brand in brands:
        profit = get_profit_per_month(brand, dataset_list)
        diff_profit[brand] = (profit[month2] - profit[month1]) / (profit[month1] + 1) * 100
    diff_profit_list = diff_profit.items()
    diff_profit_list = sorted(diff_profit_list, key=lambda x: (x[1], x[0]))
    for i in range(n):
        print(f'The brand \"{diff_profit_list[i][0]}\" lost {round(diff_profit_list[i][1] * -1, 2)}% between {month1} and {month2}.')


def top_n_two_months_losses_abs(columns:list, month1:str, month2:str, n=3):
    dataset_list = get_purchases(columns)
    brands = pd.concat(dataset_list, ignore_index=True)
    brands = brands[brands.brand.notna()].brand.unique()
    diff_profit = dict.fromkeys(brands)
    for brand in brands:
        profit = get_profit_per_month(brand, dataset_list)
        diff_profit[brand] = profit[month2] - profit[month1]
    diff_profit_list = diff_profit.items()
    diff_profit_list = sorted(diff_profit_list, key=lambda x: (x[1], x[0]))
    for i in range(n):
        print(f'The brand \"{diff_profit_list[i][0]}\" lost {round(diff_profit_list[i][1] * -1, 2):,.2f}$ between {month1} and {month2}.')

########## RQ6 ############################################################################################################

##*****hourly average of visitors for each day of the week
def havg_visit():
    #import the data of October and parse the column event_time to datetime
    data = pd.read_csv("2019-Oct.csv", #nrows=9000000, 
                   parse_dates=['event_time'],
                   date_parser=pd.to_datetime)

    #import the data of November
    data2 = pd.read_csv("2019-Nov.csv", #nrows=9000000, 
                   parse_dates=['event_time'],
                   date_parser=pd.to_datetime)

    #concatenate the two datasets
    fr=[data,data2]
    data_tot=pd.concat(fr,ignore_index=True)
    #Plot of the hourly average of visitors for each day of the week
    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.set()
    # obtain two subsets of only the views for each month
    view_ott=data[data.event_type=='view']
    view_nov=data2[data2.event_type=='view']
    #group the observations by day of the week and by hour of the day obtaining a matrix containing the numbers of views
    sett_ott=view_ott.groupby([data.event_time.dt.weekday , data.event_time.dt.hour]).size().unstack().fillna(0)
    sett_nov=view_nov.groupby([data2.event_time.dt.weekday , data2.event_time.dt.hour]).size().unstack().fillna(0)
    #obtain the average number of visitors dividing each element of the matrix by 4 or by 5 according to the frequency of each weekday in each month
    med_ott=[sett_ott[0:1]/4,sett_ott[1:4]/5,sett_ott[4:7]/4]
    med_ott=pd.concat(med_ott)
    med_nov=[sett_nov[0:4]/4,sett_nov[4:6]/5,sett_nov[6:7]/4]
    med_nov=pd.concat(med_nov)
    #plot the results in 2 subplots heatmaps where it's easy to visualize in wich part of the week the website is most visited 
    fig, axes = plt.subplots(1,2)
    fig.suptitle('Hourly average of visitors for each day of the week')
    sns.heatmap(med_ott, ax=axes[0], vmax=200000)
    axes[0].set_title('October')
    sns.heatmap(med_nov, ax=axes[1], vmax=200000)
    axes[1].set_title('November')

########## RQ7 ############################################################################################################

## ***** Prove that the pareto principle applies to your store.
def prove_pareto(columns:list):
    global month_files
    purchases = pd.concat(get_purchases(columns), ignore_index=True)
    total_business = purchases.price.sum()
    n20_users = int(purchases.user_id.nunique() * 0.2)
    profit_from_n20_users = purchases.groupby('user_id').price.agg('sum').nlargest(n20_users).sum()
    print(f'The 20% of the users is responsible of the {round(profit_from_n20_users / total_business * 100)}% of the business.')



def prove_pareto_ops(columns:list):
    global month_files
    dataset = pd.Series()
    for month in month_files:
        dataset = pd.concat([dataset, pd.read_csv(os.path.join(DATA_PATH, month), usecols=columns, dtype=np.int32, squeeze=True)], ignore_index=True)
    n20_users = int(dataset.nunique() * 0.2)
    total_ops = dataset.size
    ops_from_n20_users = dataset.value_counts().nlargest(n20_users).sum()
    print(f'The 20% of the users is responsible of the {round(ops_from_n20_users / total_ops * 100)}% of the business.')
        
