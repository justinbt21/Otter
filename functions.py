import pandas as pd
from nltk.corpus import stopwords
import re
from thefuzz import fuzz, process
import nltk
import constants
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(line):
    # remove stopwords and only include words
    clean1 = re.sub(r"[^A-Za-z\s]", "", line)
    clean2 = re.sub(r"\b(oz|ml|\w{1}|on|pc|combo|liter)\b", "", clean1)
    clean_stop = " ".join([word for word in clean2.split() if word not in stop_words])

    return clean_stop

def find_root_item_type(word, word_list, limit, min_score=90):
    _words = process.extract(word, word_list, scorer=fuzz.token_sort_ratio, limit=limit)
    score_words = list(filter(lambda x: x[1] >= min_score, _words))
    filtered_words = [x[0] for x in score_words]
    if len(filtered_words) > 1:
        filtered_words.sort(key=len)
    elif len(filtered_words) == 0:
        filtered_words = list(word)

    return filtered_words[0]

def get_rolling_amount(grp, freq, sum_col):
    return grp.rolling(freq, on='date')[sum_col].sum()

def get_rolling_df(df, grp, freq, freq_str, sum_col):
    df.sort_values(by=['date', 'hour'], inplace=True)


    df[f'{freq}_{sum_col}'] = df.groupby(grp, as_index=False, group_keys=False).apply(get_rolling_amount, freq_str, sum_col)

    return df[['date',f'{grp}',f'{sum_col}',f'{freq}_{sum_col}']]

def agg_metrics(df, groupby):

    new_df = df.groupby(groupby, as_index=False, group_keys=False).agg(constants.calc_metrics)

    new_df['acceptance_rate'] = new_df['accepted_orders'] / new_df['requested_orders']
    new_df['completion_rate'] = new_df['completed_orders'] / new_df['accepted_orders']
    new_df['order_issue_rate'] = new_df['order_issues'] / new_df['completed_orders']
    new_df['promo_order_rate'] = new_df['total_orders_promo'] / new_df['accepted_orders']
    new_df['first_time_order_rate'] = new_df['first_time_orders'] / new_df['accepted_orders']
    new_df['first_time_orders_organic'] = new_df['first_time_orders'] - new_df['first_time_orders_promo']
    new_df['returning_orders_organic'] = new_df['returning_orders'] - new_df['returning_orders_promo']
    new_df['returning_order_rate'] = new_df['returning_orders'] / new_df['accepted_orders']
    new_df['pct_first_time_promo'] = new_df['first_time_orders_promo'] / new_df['first_time_orders']
    new_df['pct_returning_promo'] = new_df['returning_orders_promo'] / new_df['returning_orders']

    return new_df

