import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Load the dataset
df = pd.read_csv("dk_news_2016_2024.csv")
print(df.head())

# check number of articles per year
df['publication_year'] = pd.to_datetime(df['published_date'], errors='coerce').dt.year
year_counts = df['publication_year'].value_counts().reset_index()
year_counts.columns = ['year', 'count']
print("\nNumber of articles per year:")
print(year_counts.sort_values('year'))
# Plot the number of articles per year
fig = px.bar(year_counts.sort_values('year'), x='year', y='count', title='Number of News Articles Published per Year in Denmark', labels={'count': 'Number of Articles', 'year': 'Year'})
fig.show()

# check publisher ratio per year
publisher_year_counts = df.groupby(['publication_year', 'publisher']).size().reset_index(name='count')
print("\nPublisher counts per year:")
print(publisher_year_counts)
# Plot publisher ratio per year
fig2 = px.bar(publisher_year_counts, x='publication_year', y='count', color='publisher', title='Publisher Distribution of News Articles per Year in Denmark', labels={'count': 'Number of Articles', 'publication_year': 'Year', 'publisher': 'Publisher'})
fig2.show()
# AI news ratio of Publisher Distribution of News Articles per Year in Denmark
# first, mark ai related articles


# find 10 random articles and print their title, published_date, select only from publishers sn.dk
sn_articles = df[(df['publisher'] == 'sn.dk')&(df['published_date']>='20220101')].sample(10, random_state=42)
print("\n10 Random Articles from sn.dk:")
print(sn_articles[['title', 'published_date']])

# check duplicated articles based on title and publisher
duplicated_articles = df[df.duplicated(subset=['title', 'publisher'], keep=False)]
# publishers with dulicated articles and their counts
duplicated_counts = duplicated_articles['publisher'].value_counts().reset_index()
duplicated_counts.columns = ['publisher', 'duplicated_count']
print("\nDuplicated Articles Counts by Publisher:")
print(duplicated_counts)
# plot duplicated articles counts ratio(duplicated/total) articles by publisher
total_counts = df['publisher'].value_counts().reset_index()
total_counts.columns = ['publisher', 'total_count']
merged_counts = pd.merge(duplicated_counts, total_counts, on='publisher', how='left')
merged_counts['duplicated_ratio'] = merged_counts['duplicated_count'] / merged_counts['total_count']
fig3 = px.bar(merged_counts, x='publisher', y='duplicated_ratio', title='Duplicated Articles Ratio by Publisher in Denmark', labels={'duplicated_ratio': 'Duplicated Articles Ratio', 'publisher': 'Publisher'})
fig3.show()

# https://www.sn.dk/art6462880