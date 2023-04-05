import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from functions import agg_metrics
from constants import food_types, metrics
from wordcloud import WordCloud

st.set_page_config(layout="centered")
# Read DataFrame
df = pd.read_csv('data.csv')

# Sidebar Global Widgets
st.sidebar.title('Global Widgets')
st.sidebar.caption(':red[(Will affect all data displayed)]')
type_toggle = st.sidebar.radio("Toggle between Food Type", pd.Series(food_types.keys()))
metrics_toggle = st.sidebar.selectbox("Choose between Metrics displayed", pd.Series((metrics.keys())))
topn_input = st.sidebar.slider('Top Items Selector', min_value=0, max_value=50, value=15, step=5)
#min_rating = st.sidebar.radio('Min Rating (TEST)', pd.Series([0,1,2,3,4,5]), horizontal=True)
#max_order_issue_rate = st.sidebar.slider('Max Order Issue Rate (TEST)', 0.0, 1.0, 1.0, .1)
exclude_cuisine = st.sidebar.multiselect('Exclude specific cuisine tags', df['tags'].drop_duplicates())
include_cuisine = st.sidebar.multiselect('Only include specific cuisine tags', df['tags'].drop_duplicates())
dupe_cuisine = list(set(exclude_cuisine).intersection(include_cuisine))
if len(dupe_cuisine) > 0:
    st.sidebar.error(f"{dupe_cuisine} are in both exclude and include. Please remove them from one")

# Data Tranformations
if len(include_cuisine) > 0:
    df = df.loc[(~df.tags.isin(exclude_cuisine)) & (df.tags.isin(include_cuisine))]
else:
    df = df.loc[~df.tags.isin(exclude_cuisine)]
dates = (pd.to_datetime(df['date_str']).drop_duplicates())
df.sort_values(by=['date', 'hour'], inplace=True)
df.reset_index(drop=True, inplace=True)
# Used to identify largest cohorts for graph
largest_cohorts = agg_metrics(df, food_types[type_toggle]) \
    .nlargest(topn_input, 'requested_orders')[food_types[type_toggle]].drop_duplicates()


# Start of App
st.title('Otter Takehome - Food Analytics')
st.header(f'Day Of Week - Trending {type_toggle}')
# Day of Week Data
all_fig = px.line(agg_metrics(df.loc[df[food_types[type_toggle]].isin(largest_cohorts)], ['dayofweek',food_types[type_toggle]]).sort_values(by='dayofweek')
                        , x='dayofweek', y=metrics[metrics_toggle], color=food_types[type_toggle],)
st.plotly_chart(all_fig)

st.subheader('Day of Week - Deep Dive')
dow_input = st.selectbox('What day of week would you like to look at?',
                         options=df['dayname'].drop_duplicates())
st.subheader(f'Top {type_toggle} on {dow_input} by {metrics_toggle}')
dow_df_prep = df.loc[df.dayname == dow_input]
dow_df = agg_metrics(dow_df_prep, [food_types[type_toggle], 'dayname', 'dayofweek']) \
    .nlargest(topn_input, metrics[metrics_toggle]) \
    .sort_values(by=metrics[metrics_toggle], ascending=False).reset_index(drop=True)

st.plotly_chart(px.bar(dow_df, x=food_types[type_toggle], y=metrics[metrics_toggle],
                       labels={metrics[metrics_toggle]: metrics_toggle,
                               food_types[type_toggle]: type_toggle}
                       )
                )
# Dropdown Table to see per cuisine, what food items are selling
if type_toggle == 'Cuisine Type':
    with st.expander('Click here to deep dive into cuisines'):
        cuisine_list = dow_df[food_types[type_toggle]].drop_duplicates()
        cuisine_input = st.selectbox('Select a cuisine to see which food items are popular', options=cuisine_list)
        mask = dow_df_prep['tags'].apply(lambda x: x == cuisine_input)
        st.dataframe(
            agg_metrics(dow_df_prep.loc[mask], 'item_type_new') \
                .sort_values(by=metrics[metrics_toggle], ascending=False).reset_index(drop=True)
            )

st.dataframe(agg_metrics(dow_df_prep, food_types[type_toggle])
             .sort_values(by=metrics[metrics_toggle], ascending=False).reset_index(drop=True)
             )



# Time of Day Data
st.header(f'Time of Day - Trending {type_toggle}')
# High Level Graph
st.plotly_chart(px.line(agg_metrics(df.loc[df[food_types[type_toggle]].isin(largest_cohorts)], ['hour',food_types[type_toggle]]).sort_values(by='hour')
                        , x='hour', y=metrics[metrics_toggle], color=food_types[type_toggle]))
st.subheader(f'Time of Day - Deep Dive')
toggle_hours = st.radio('Filter between:', pd.Series(['Exact Hour','Timeframe']), horizontal=True)
if toggle_hours == 'Exact Hour':
    # Included 24 to add exclusivity on right contraint
    hour_input = st.slider('Hour of Day', min_value=0, max_value=24, value=0, step=1)
    st.subheader(f'{metrics_toggle} at {hour_input}:00')
    order_hour_prep = df.loc[df.hour == hour_input]
elif toggle_hours == 'Timeframe':
    beg_hr, end_hr = st.select_slider('Hour of Day', range(0,25,1), value=(0,24))
    st.subheader(f'{metrics_toggle} between {beg_hr}:00 and {end_hr}:00')
    order_hour_prep = df.loc[(df.hour >= beg_hr) & (df.hour < end_hr)]


# Hourly Data
order_hour_df = agg_metrics(order_hour_prep, [food_types[type_toggle], 'hour']) \
    .sort_values(by=metrics[metrics_toggle], ascending=False).reset_index(drop=True)

fig_hr_df = agg_metrics(order_hour_df, food_types[type_toggle]) \
    .nlargest(topn_input, metrics[metrics_toggle]) \
    .sort_values(by=metrics[metrics_toggle], ascending=False).reset_index(drop=True) \
    .reset_index(drop=True)
st.write(px.bar(fig_hr_df, x=food_types[type_toggle], y=metrics[metrics_toggle],
                labels={metrics[metrics_toggle]: metrics_toggle,
                        food_types[type_toggle]: type_toggle},
                )
         )
if type_toggle == 'Cuisine Type':
    with st.expander('Click here to deep dive into cuisines'):
        cuisine_list = order_hour_df[food_types[type_toggle]].drop_duplicates()
        cuisine_input = st.selectbox('Select a cuisine to see which food items are popular', options=cuisine_list)
        mask = order_hour_prep['tags'].apply(lambda x: x == cuisine_input)
        st.dataframe(agg_metrics(order_hour_prep.loc[mask], 'item_type_new') \
                     .sort_values(by=metrics[metrics_toggle], ascending=False).reset_index(drop=True)
                     )
st.dataframe(fig_hr_df)

with st.expander('Appendix'):
    #Order Rate
    subfig = make_subplots(specs=[[{"secondary_y": True}]])
    fig1 = px.line(agg_metrics(df.loc[df[food_types[type_toggle]].isin(largest_cohorts)], ['dayofweek']).sort_values(by='dayofweek')
                            , x='dayofweek', y=['promo_order_rate', 'first_time_order_rate'])
    fig2 = px.line(agg_metrics(df.loc[df[food_types[type_toggle]].isin(largest_cohorts)], ['dayofweek']).sort_values(by='dayofweek')
                            , x='dayofweek', y=['requested_orders'])
    fig2.update_traces(yaxis="y2", line_color='red')
    subfig.add_traces(fig1.data + fig2.data)
    subfig.layout.xaxis.title = "Day of Week"
    subfig.layout.yaxis.title = "Rate"
    subfig.layout.yaxis2.title = "Orders"
    subfig.update_layout(title='Requested Orders relative to Promo Orders')
    st.plotly_chart(subfig)

    breakfast_df = agg_metrics(df.loc[(df.hour >= 5) & (df.hour < 11)], 'item_type_new')[['item_type_new','requested_orders']].set_index('item_type_new').to_dict()
    lunch_df = agg_metrics(df.loc[(df.hour >= 11) & (df.hour < 17)], 'item_type_new')[['item_type_new','requested_orders']].set_index('item_type_new').to_dict()
    dinner_df = agg_metrics(df.loc[(df.hour >= 17) & (df.hour < 23)], 'item_type_new')[['item_type_new','requested_orders']].set_index('item_type_new').to_dict()
    late_night_df = agg_metrics(df.loc[(df.hour >= 23) | (df.hour < 5)], 'item_type_new')[['item_type_new','requested_orders']].set_index('item_type_new').to_dict()
    # TIme of Day - Food Item Word Cloud
    breakfast_wc = WordCloud().fit_words(breakfast_df['requested_orders'])
    lunch_wc = WordCloud().fit_words(lunch_df['requested_orders'])
    dinner_wc = WordCloud().fit_words(dinner_df['requested_orders'])
    late_night_wc = WordCloud().fit_words(late_night_df['requested_orders'])

    st.image(breakfast_wc.to_array())
    st.image(lunch_wc.to_array())
    st.image(dinner_wc.to_array())
    st.image(late_night_wc.to_array())

    breakfast_df = agg_metrics(df.loc[(df.hour >= 5) & (df.hour < 11)], 'tags')[['tags','requested_orders']].set_index('tags').to_dict()
    lunch_df = agg_metrics(df.loc[(df.hour >= 11) & (df.hour < 17)], 'tags')[['tags','requested_orders']].set_index('tags').to_dict()
    dinner_df = agg_metrics(df.loc[(df.hour >= 17) & (df.hour < 23)], 'tags')[['tags','requested_orders']].set_index('tags').to_dict()
    late_night_df = agg_metrics(df.loc[(df.hour >= 23) | (df.hour < 5)], 'tags')[['tags','requested_orders']].set_index('tags').to_dict()
    # Time of Day - Cuisine Word Cloud
    breakfast_wc2 = WordCloud().fit_words(breakfast_df['requested_orders'])
    lunch_wc2 = WordCloud().fit_words(lunch_df['requested_orders'])
    dinner_wc2 = WordCloud().fit_words(dinner_df['requested_orders'])
    late_night_wc2 = WordCloud().fit_words(late_night_df['requested_orders'])

    st.image(breakfast_wc2.to_array())
    st.image(lunch_wc2.to_array())
    st.image(dinner_wc2.to_array())
    st.image(late_night_wc2.to_array())