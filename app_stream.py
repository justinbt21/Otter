import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from functions import agg_metrics
from pandarallel import pandarallel
from constants import metrics, food_types, calc_metrics

pandarallel.initialize()
st.set_page_config(layout="centered")
# Read DataFrame
df = pd.read_csv('data.csv')

# Sidebar Global Widgets
st.sidebar.title('Global Widgets')
st.sidebar.caption(':red[(Will affect all data displayed)]')
type_toggle = st.sidebar.radio("Toggle between Food Type", pd.Series(food_types.keys()))
#toggle_data = st.sidebar.checkbox('Use predicted labels', value=False)
#if toggle_data:
#    df = pd.read_csv('pred_data.csv')
#else:
#    df = pd.read_csv('data.csv')
metrics_toggle = st.sidebar.selectbox("Choose between Metrics displayed", pd.Series((metrics.keys())))
topn_input = st.sidebar.slider('Top Items Selector', min_value=0, max_value=50, value=15, step=5)
min_rating = st.sidebar.radio('Min Rating (TEST)', pd.Series([0,1,2,3,4,5]), horizontal=True)
max_order_issue_rate = st.sidebar.slider('Max Order Issue Rate (TEST)', 0.0, 1.0, 1.0, .1)
exclude_cuisine = st.sidebar.multiselect('Exclude specific cuisine tags', df['tags'].drop_duplicates())
df = df.loc[~df.tags.isin(exclude_cuisine)]

# Data Tranformations
dates = (pd.to_datetime(df['date_str']).drop_duplicates())
df.sort_values(by=['date', 'hour'], inplace=True)
df.reset_index(drop=True, inplace=True)
# Used to identify largest cohorts for graph
largest_cohorts = agg_metrics(df, food_types[type_toggle]) \
    .nlargest(topn_input, 'requested_orders')[food_types[type_toggle]].drop_duplicates()


# Start of App
st.title('Otter Takehome - Food Analytics')
st.subheader('Day Of Week - Trending Items')
dow_input = st.selectbox('What day of week would you like to look at?',
                         options=df['dayname'].drop_duplicates())


# Day of Week Data
all_fig = px.line(agg_metrics(df.loc[df[food_types[type_toggle]].isin(largest_cohorts)], ['dayofweek',food_types[type_toggle]]).sort_values(by='dayofweek')
                        , x='dayofweek', y=metrics[metrics_toggle], color=food_types[type_toggle],)
st.plotly_chart(all_fig)
st.subheader(f'Top {type_toggle} on {dow_input} by {metrics_toggle}')
dow_df_prep = df.loc[df.dayname == dow_input]
dow_df = agg_metrics(dow_df_prep, [food_types[type_toggle], 'dayname', 'dayofweek']) \
    .nlargest(topn_input, metrics[metrics_toggle]) \
    .sort_values(by=metrics[metrics_toggle], ascending=False).reset_index(drop=True)
#dow_df = dow_df_prep.groupby([food_types[type_toggle], 'dayname','dayofweek'], as_index=False, group_keys=False)[
#    metrics[metrics_toggle]].sum() \
#    .nlargest(topn_input, metrics[metrics_toggle]) \
#    .sort_values(by=metrics[metrics_toggle], ascending=False).reset_index(drop=True)


st.plotly_chart(px.bar(dow_df, x=food_types[type_toggle], y=metrics[metrics_toggle],
                       labels={metrics[metrics_toggle]: metrics_toggle,
                               food_types[type_toggle]: type_toggle}
                       )
                )

st.dataframe(agg_metrics(dow_df_prep, food_types[type_toggle])
             .sort_values(by=metrics[metrics_toggle], ascending=False).reset_index(drop=True)
             )
if type_toggle == 'Cuisine Type':
    with st.expander('Click here to deep dive into cuisines'):
        cuisine_list = dow_df[food_types[type_toggle]].drop_duplicates()
        cuisine_input = st.selectbox('Select a cuisine to see which food items are popular', options=cuisine_list)
        mask = dow_df_prep['tags'].apply(lambda x: x == cuisine_input)
        st.dataframe(
            agg_metrics(dow_df_prep.loc[mask], 'item_type_new') \
                .sort_values(by=metrics[metrics_toggle], ascending=False).reset_index(drop=True)
            )

# Time of Day Data
st.subheader('Time of Day - Trending Items')
toggle_hours = st.radio('Filter between:', pd.Series(['Exact Hour','Timeframe']), horizontal=True)
if toggle_hours == 'Exact Hour':
    # Included 24 to add exlusivity on right contraint
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

st.dataframe(fig_hr_df)
if type_toggle == 'Cuisine Type':
    with st.expander('Click here to deep dive into cuisines'):
        cuisine_list = order_hour_df[food_types[type_toggle]].drop_duplicates()
        cuisine_input = st.selectbox('Select a cuisine to see which food items are popular', options=cuisine_list)
        mask = order_hour_prep['tags'].apply(lambda x: x == cuisine_input)
        st.dataframe(agg_metrics(order_hour_prep.loc[mask], 'item_type_new') \
                     .sort_values(by=metrics[metrics_toggle], ascending=False).reset_index(drop=True)
                     )


st.plotly_chart(px.line(agg_metrics(df.loc[df[food_types[type_toggle]].isin(largest_cohorts)], ['hour',food_types[type_toggle]]).sort_values(by='hour')
                        , x='hour', y=metrics[metrics_toggle], color=food_types[type_toggle]))


st.dataframe(agg_metrics(df.loc[df[food_types[type_toggle]].isin(largest_cohorts)], food_types[type_toggle]))

# df[['dayofweek','dayname']].drop_duplicates()


st.header('Appendix')
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


import plotly.graph_objects as go

x=['1/11', '1/12', '1/13', '1/14']

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=[40, 60, 40, 10],
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='rgb(131, 90, 241)'),
    stackgroup='one' # define stack group
))
fig.add_trace(go.Scatter(
    x=x, y=[20, 10, 10, 60],
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='rgb(111, 231, 219)'),
    stackgroup='one'
))
fig.add_trace(go.Scatter(
    x=x, y=[40, 30, 50, 30],
    hoverinfo='x+y',
    mode='lines',
    line=dict(width=0.5, color='rgb(184, 247, 212)'),
    stackgroup='one'
))

fig.update_layout(yaxis_range=(0, 100))
st.plotly_chart(fig)

st.write(df.cols)