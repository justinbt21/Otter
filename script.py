import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import warnings
from sklearn.metrics import hamming_loss
from pandarallel import pandarallel
from thefuzz import process, fuzz
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MultiLabelBinarizer

pandarallel.initialize()
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def find_root_item_type(word, word_list, limit, min_score=90):
    _words = process.extract(word, word_list, scorer=fuzz.token_sort_ratio, limit=limit)
    score_words = list(filter(lambda x: x[1] >= min_score, _words))
    score_words.sort(key=lambda x: len(x[0]) / (x[1]))
    filtered_words = [x[0] for x in score_words]
    if len(filtered_words) > 1:
        filtered_word = filtered_words[0]
    else:
        filtered_word = word

    return filtered_word

def clean_text(line):
    clean1 = re.sub(r"\b(oz|ml|(\d\w)|on|pc|combo|liter)\b|(\(.+\))|[^A-Za-z\s]", "", line)
    #    clean2=re.sub(r"", "", clean1)
    clean_stop = " ".join([word for word in clean1.split() if word not in stop_words])

    return clean_stop

raw_df = pd.read_csv('./raw.csv')

df = raw_df.copy()
df['total_eater_revenue'] = df[['total_eater_spend','total_eater_discount']].sum(axis=1)
df['total_orders_promo'] = df['first_time_orders_promo'] + df['returning_orders_promo']
df['completion_rate'] = df['completed_orders'] / df['accepted_orders']
df['acceptance_rate'] = df['accepted_orders'] / df['requested_orders']
df['order_issue_rate'] = df['order_issues'] / df['completed_orders']
df['first_time_orders_organic'] = df['first_time_orders'] - df['first_time_orders_promo']
df['returning_orders_organic'] = df['returning_orders'] - df['returning_orders_promo']
df['first_time_order_rate'] = df['first_time_orders'] / df['accepted_orders']
df['returning_order_rate'] = df['returning_orders'] / df['accepted_orders']
df['avg_prep_time_min'] = df['avg_prep_time'] / 60.0
df['spend_per_prep_min'] = df['total_eater_spend'] / (df['avg_prep_time']*df['accepted_orders']*1.0)
df['total_eater_revenue'] = df[['total_eater_spend','total_eater_discount']].sum(axis=1)
df['date_str'] = df['date'].copy()
df['date'] = pd.to_datetime(df['date'])
df['dayofweek'] = df['date'].dt.dayofweek
df['dayname'] = df['date'].dt.day_name()

# preprocess text by removing stop words
# Decided against stemming due to poor performance
stop_words = set(stopwords.words('english'))

df['name'] = df['name'].str.lower()
df['clean_name'] = df['name'].apply(clean_text)

desc_regex= r"(\b(spicy|^classic|ultimate|signature)\b)|(french|fresh cut|home)\s(?<!fries)"
df['item_type'] = df['clean_name'].apply(lambda x: " ".join(w for w in re.sub(desc_regex, "", x).split())).tolist()
df.loc[df.item_type == '', 'item_type'] = df['clean_name']
all_items = set(df['item_type'].tolist())

# create food item dictionary
item_df = df[['item_type']].drop_duplicates()
# Fuzzy match dictionary on self to consolidate item types
item_df['item_type_new'] = item_df.parallel_apply(lambda x: find_root_item_type(x['item_type'], all_items, min_score=90, limit=6), axis=1)

df2 = df.merge(item_df, on='item_type')

cuisine_t={
    'italian':r'fettucini|rigatoni|lasagna|spaghetti|penne|gnocchi|tortellini|pasta|carbonara|pizza|calzone|garlic bread|alfredo|mozzarella|caesar|cacio|fe[t]{1,2}u[c]{1,2}ine|ravioli|burrata|proscuito|chicken parm|alfredo',
    'vietnamese': r'\b(pho)\b|spring roll|vietnamese|\b(ba[nh]{2})\b mi|thit nuong|cha gio|\bcuon\b|summer roll',
    'korean': r'bibimbap|korean|kimchi',
    'indian': r'paneer|tikka|masala|indian|pakora|gobhi|samosa|naan|basmati|lassi|saag|biryani|makhni|vindaloo|tandoori|korma|butter chicken|dolma',
    'southern': r'fried chicken|gumbo|brisket|smoke|bbq|fried zucchini|bbq|coleslaw',
    'mediterranean': r'pita|tabouleh|fattoush|gyro|kebab|kabob|skewer|falafel|greek|kofta|shawarma|hummus|tzatziki',
    'breakfast': r'egg|breakfast|bagel|toast|bacon|omelette|hash brown|croissant|lox|waffle|pancake|sausage',
    'american': r'mac.*cheese|burger|\bwing[s]?\b|bacon|reuben|cheesesteak|tater tots|fries|buffalo|ranch|onion rings|grilled cheese|melt|nashville|slider|chili cheese|garlic knots|tender|nuggets|dog',
    'chinese': r'orange chicken|tofu|chinese|mein|dumplings|mongolian|potsticker|fried rice|general tsos|wontons|chow fun|szechuan|beef broccoli|kung pao|\b(beef broccoli)\b',
    'japanese': r'ramen|sushi|sashimi|nigiri|unagi|katsu|((?<!egg)(?<!spring)(?<!lobster)(?<!lamb)(?<!curry)(?<!cinnamon)\sroll)|gyoza|tempura|miso|edamame|udon|wasabi|karaage|teriyaki|\bsoba\b',
    'mexican': r'mexican|taco|burrito|guac|chorizo|al pastor|quesadilla|salsa|birria|horchata|carne asada|el verde|refried beans|tostada|nachos|churro|tortillas',
    'latin': r'\barepa\b|empanada|jerk|caribbean',
    'thai': r'panang|pad thai|pad see ew|\bthai\b|drunken noodle|((red|yellow|green)\scurry)|tom kha|massaman|satay',
    'sandwiches': r'sandwich|blt|turkey club|roast beef',
    'soup': r'(soup)',
    'coffee': r'latte|capuccino|coffee|cappucino|cold brew',
    'drinks': r'water|coke|sprite|ginger ale|lemonade|pepsi|juice|\b(tea)\b|gatorade',
    'hawaiian': r'hawaiian|poke|musubi',
    'healthy': r'salad|juice|healthy|fruit|acai|berry|vegan|vegetables|veggies|smoothie',
    'sweets': r'waffle|ice cream|tiramisu|oreo|cinnamon roll|cheesecake|smoothie|donuts|chocolate|cookie|caramel|pudding',
    'seafood': r'fish|lobster|crab|shrimp',
    'rice': r'rice bowl|white rice',
}


cuisine_list = list(cuisine_t.keys())
for k,v in cuisine_t.items():
    df2.loc[df2['item_type_new'].str.contains(v, regex=True), k] = 1

df2[cuisine_list] = df2[cuisine_list].fillna(0)
df2['sum'] = df2[cuisine_list].sum(axis=1)
df2.loc[df2['sum'] == 0].shape

df2['tags'] = df2[cuisine_list].gt(0).apply(lambda x: x.index[x].tolist(), axis=1)
df2.loc[df2.item_type_new.str.contains('wing')][['clean_name','item_type','item_type_new','tags']]

valid_df = df2.loc[df2['sum'] >= 1]
unlabel_df = df2.loc[df2['sum'] == 0].reset_index(drop=True)
unlabel_df['tags'] = unlabel_df[list(cuisine_t.keys())].gt(0).apply(lambda x: x.index[x].tolist(), axis=1)

xgb_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(strip_accents='unicode', analyzer='word', max_features=200)),
    ('clf', OneVsRestClassifier(GradientBoostingClassifier()))
])

mlb = MultiLabelBinarizer(classes=cuisine_list)
mlb.fit(cuisine_list)

X = valid_df['item_type_new']
y = mlb.transform(valid_df['tags'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=1)

model_list = {'XGB': xgb_pipe}  # ,'Logistic': logi_pipe,'Naive Bayes': nb_pipe}

for k, v in model_list.items():
    print(f"Fitting {k} Model to data")
    v.fit(x_train, y_train)

    test_predict = v.predict(x_test)
    train_predict = v.predict(x_train)
    print(
        f'{k} F1 for train for is {classification_report(y_train, train_predict, target_names=list(cuisine_t.keys()))}')
    print(f'{k} F1 for test for is {classification_report(y_test, test_predict, target_names=list(cuisine_t.keys()))}')
    print(f'{k} Hamming Loss is {hamming_loss(y_test, test_predict)}')

ux = unlabel_df['item_type_new']
uy = mlb.transform(unlabel_df['tags'])

y_pred = xgb_pipe.predict(ux)
y_pred_tags = mlb.inverse_transform(y_pred)

y_tags = pd.Series(map(list, y_pred_tags), name='pred_tags')
pred_df = unlabel_df.merge(y_tags, left_index=True, right_index=True)

for i in cuisine_list:
    pred_df[i] = pred_df.apply(lambda x: 1 if i in x['pred_tags'] else 0, axis = 1)

pred_df['sum'] = pred_df[cuisine_list].sum(axis=1)

final_df = pd.concat([valid_df, unlabel_df])
final_pred_df = pd.concat([valid_df, pred_df])
if (final_df.shape[0] == raw_df.shape[0]) & (final_pred_df.shape[0] == raw_df.shape[0]):
    final_df.to_csv('data.csv', header=True, index=False)
    final_pred_df.to_csv('pred_data.csv', header=True, index=False)
else:
    raise ValueError('Rows didnt match')