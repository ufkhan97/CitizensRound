import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import plotly.graph_objs as go
import plotly.express as px
import locale
import networkx as nx
import time


st.set_page_config(
    page_title="Gitcoin Citizens Round",
    page_icon="📊",
    layout="wide",

)

st.title('Gitcoin Citizens Rounds')

dfpp = pd.read_csv('citizens_round_passports.csv')
df = pd.read_csv('citizens_round_votes.csv')

st.write('Number of projects: ', df['project_id'].nunique())
st.write('Number of voters: ', df['voter'].nunique())
st.write('Total amountUSD: ', df['amountUSD'].sum())
st.write('Number of votes: ', df['amountUSD'].count())
st.write('Number of passports: ', dfpp['address'].nunique())

df['voter'] = df['voter'].str.lower()
df = pd.merge(df, dfpp[['address', 'rawScore']], how='left', left_on='voter', right_on='address')
df = df.rename(columns={'rawScore': 'passport_score'})
df['passport_score'] = df['passport_score'].fillna(0)
# convert passport_score to float
df['passport_score'] = df['passport_score'].astype(float)

# Calculate the number of unique voter addresses for each floor of raw score

df['floor_passport_score'] = df['passport_score'].apply(np.floor)
unique_voters = df.groupby('floor_passport_score')['voter'].nunique().reset_index()

grants_color = 'blue'
voters_color = 'red'
line_color = '#008F11'

# Minimum donation amount to include, start at 10
min_donation = st.slider('Minimum donation amount', value=10, max_value=1000, min_value=1, step=1)

# Filter the dataframe to include only rows with donation amounts above the threshold
df = df[df['amountUSD'] > min_donation]

# Minimum passport score to include, start at 10
min_passport_score = st.slider('Minimum Passport Score', value=15, max_value=100, min_value=1, step=1)

# Filter the dataframe to include only rows with donation amounts above the threshold
df = df[df['passport_score'] > min_passport_score]

# Maximum Block Number to include, start at Min
max_block_number = st.slider('Maximum Block Number', value=df['blockNumber'].max(), max_value=df['blockNumber'].max(), min_value=df['blockNumber'].min(), step=100)

# Filter to include only rows with blockNumber below the threshold
df = df[df['blockNumber'] < max_block_number]

# Initialize a new Graph
B = nx.Graph()

# Create nodes with the bipartite attribute
B.add_nodes_from(df['voter'].unique(), bipartite=0, color=voters_color) 
B.add_nodes_from(df['title'].unique(), bipartite=1, color=grants_color) 



# Add edges with amountUSD as an attribute
for _, row in df.iterrows():
    B.add_edge(row['voter'], row['title'], amountUSD=row['amountUSD'])



# Compute the layout
current_time = time.time()
pos = nx.spring_layout(B, dim=3, k = .09, iterations=50)
new_time = time.time()


    
# Extract node information
node_x = [coord[0] for coord in pos.values()]
node_y = [coord[1] for coord in pos.values()]
node_z = [coord[2] for coord in pos.values()] # added z-coordinates for 3D
node_names = list(pos.keys())
# Compute the degrees of the nodes 
degrees = np.array([B.degree(node_name) for node_name in node_names])
# Apply the natural logarithm to the degrees 
log_degrees = np.log(degrees + 1)
node_sizes = log_degrees * 10

# Extract edge information
edge_x = []
edge_y = []
edge_z = []  
edge_weights = []

for edge in B.edges(data=True):
    x0, y0, z0 = pos[edge[0]]
    x1, y1, z1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])
    edge_z.extend([z0, z1, None])  
    edge_weights.append(edge[2]['amountUSD'])

# Create the edge traces
edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z, 
    line=dict(width=1, color=line_color),
    hoverinfo='none',
    mode='lines',
    marker=dict(opacity=0.5))


# Create the node traces
node_trace = go.Scatter3d(
    x=node_x, y=node_y, z=node_z,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        color=[data['color'] for _, data in B.nodes(data=True)],  # color is now assigned based on node data
        size=node_sizes,
        opacity=1,
        sizemode='diameter'
    ))


node_adjacencies = []
for node, adjacencies in enumerate(B.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))
node_trace.marker.color = [data[1]['color'] for data in B.nodes(data=True)]


# Prepare text information for hovering
node_trace.text = [f'{name}: {adj} connections' for name, adj in zip(node_names, node_adjacencies)]

# Create the figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='3D Network graph of voters and grants',
                    titlefont=dict(size=20),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=False,
                        text="This graph shows the connections between voters and grants based on donation data.",
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002 )],
                    scene = dict(
                        xaxis_title='X Axis',
                        yaxis_title='Y Axis',
                        zaxis_title='Z Axis')))
                        
st.plotly_chart(fig, use_container_width=True)
