import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('filtered data/forest-cover-v1.csv')

years = [str(year) for year in range(1990, 2021)]
df['Forest Cover Change (1990-2020)'] = (df['Forest Area 2020'] - df['Forest Area 1990']).round(2)
latest_year = 'Forest Area 2020'

for year_col in [f'Forest Area {year}' for year in range(1990, 2021)]:
    df[year_col] = df[year_col].round(2)

# Create the Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("World Forest Cover Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='year-selector',
                options=[{'label': year.replace('Forest Area ', ''), 'value': year} 
                        for year in ['Forest Area 1990'] + years[1::5] + ['Forest Area 2020']],
                value='Forest Area 2020',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Dropdown(
                id='metric-selector',
                options=[
                    {'label': 'Forest Cover %', 'value': 'cover'},
                    {'label': 'Forest Cover Change (1990-2020)', 'value': 'change'},
                    {'label': 'Forest Cover per Capita', 'value': 'per_capita'}
                ],
                value='cover',
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.RadioItems(
                id='map-type',
                options=[
                    {'label': '2D Map', 'value': '2d'},
                    {'label': '3D Globe', 'value': '3d'}
                ],
                value='2d',
                inline=True,
                style={'width': '100%'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
    
    html.Div([
        dcc.Graph(
            id='world-map',
            style={'height': '600px', 'width': '100%'}
        )
    ], style={'width': '100%', 'marginBottom': '20px'}),
    
    html.Div([
        html.Div([
            dcc.Graph(
                id='top-countries',
                style={'height': '400px'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            dcc.Graph(
                id='bottom-countries',
                style={'height': '400px'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'float': 'right'}),
    ], style={'width': '100%', 'marginBottom': '20px'}),
    
    html.Div([
        dcc.Graph(
            id='time-series',
            style={'height': '400px'}
        )
    ], style={'width': '100%', 'marginBottom': '20px'}),
    
    html.Div([
        html.P("Data Source: Forest Cover Dataset", style={'textAlign': 'center', 'fontSize': 12})
    ])
], style={'maxWidth': '1200px', 'margin': '0 auto'})

@app.callback(
    [Output('world-map', 'figure'),
     Output('top-countries', 'figure'),
     Output('bottom-countries', 'figure'),
     Output('time-series', 'figure')],
    [Input('year-selector', 'value'),
     Input('metric-selector', 'value'),
     Input('map-type', 'value')]
)
def update_figures(selected_year, selected_metric, map_type):
    
    if selected_metric == 'cover':
        color_column = selected_year
        title_suffix = f" ({selected_year.replace('Forest Area ', '')})"
        color_scale = 'Greens'
        hover_template = '%{hovertext}<br>Forest Cover: %{z:.2f}%'
    elif selected_metric == 'change':
        color_column = 'Forest Cover Change (1990-2020)'
        title_suffix = " Change (1990-2020)"
        color_scale = 'RdYlGn'
        hover_template = '%{hovertext}<br>Change: %{z:.2f}%'
    else:  
        
        df['Forest per Capita'] = ((df[selected_year] * df['Area (km²)'] * 100) / 
                                  (df['Area (km²)'] * df['Population Density (per km²)'])).round(2)
        color_column = 'Forest per Capita'
        title_suffix = f" per Capita ({selected_year.replace('Forest Area ', '')})"
        color_scale = 'Blues'
        hover_template = '%{hovertext}<br>Forest per Capita: %{z:.2f} ha'
    
    # Create world map
    if map_type == '2d':
        fig_map = px.choropleth(
            df,
            locations="Country Code",
            color=color_column,
            hover_name="Country Name",
            color_continuous_scale=color_scale,
            projection="natural earth",
            title=f"World Forest Cover{title_suffix}",
            height=600
        )
        fig_map.update_traces(
            hovertemplate=hover_template
        )
        fig_map.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            coloraxis_colorbar=dict(
                title='%' if selected_metric != 'per_capita' else 'Hectares'
            )
        )
    else:
        fig_map = go.Figure(
            go.Choropleth(
                locations=df["Country Code"],
                z=df[color_column],
                colorscale=color_scale,
                autocolorscale=False,
                marker_line_color='darkgray',
                marker_line_width=0.5,
                colorbar_title='%' if selected_metric != 'per_capita' else 'Hectares',
                hovertext=df["Country Name"],
                hovertemplate=hover_template
            )
        )
        
        fig_map.update_geos(
            projection_type="orthographic",
            landcolor="lightgray",
            oceancolor="lightblue",
            showocean=True,
            lakecolor="lightblue"
        )
        
        fig_map.update_layout(
            title_text=f"World Forest Cover{title_suffix} (3D Globe)",
            height=600,
            margin={"r":0,"t":40,"l":0,"b":0},
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='orthographic'
            )
        )
    
    # Create top countries bar chart
    top_countries = df.nlargest(10, color_column)
    fig_top = px.bar(
        top_countries,
        x='Country Name',
        y=color_column,
        title=f"Top 10 Countries by Forest Cover{title_suffix}",
        color=color_column,
        color_continuous_scale=color_scale,
        labels={color_column: 'Forest Cover %' if selected_metric == 'cover' else 
               ('Change (%)' if selected_metric == 'change' else 'Hectares per person')},
        height=400
    )
    fig_top.update_traces(
        hovertemplate='%{x}<br>%{y:.2f}' + ('%' if selected_metric != 'per_capita' else ' ha')
    )
    
    # Create bottom countries bar chart
    bottom_countries = df.nsmallest(10, color_column)
    fig_bottom = px.bar(
        bottom_countries,
        x='Country Name',
        y=color_column,
        title=f"Bottom 10 Countries by Forest Cover{title_suffix}",
        color=color_column,
        color_continuous_scale=color_scale,
        labels={color_column: 'Forest Cover %' if selected_metric == 'cover' else 
               ('Change (%)' if selected_metric == 'change' else 'Hectares per person')},
        height=400
    )
    fig_bottom.update_traces(
        hovertemplate='%{x}<br>%{y:.2f}' + ('%' if selected_metric != 'per_capita' else ' ha')
    )
    
    # Create time series for selected countries (top 5)
    top_5 = df.nlargest(5, color_column)['Country Name'].tolist()
    time_series_df = df[df['Country Name'].isin(top_5)]
    time_series_df = time_series_df.melt(
        id_vars=['Country Name', 'Country Code'],
        value_vars=[f'Forest Area {year}' for year in range(1990, 2021)],
        var_name='Year',
        value_name='Forest Cover'
    )
    time_series_df['Year'] = time_series_df['Year'].str.replace('Forest Area ', '').astype(int)
    
    fig_time = px.line(
        time_series_df,
        x='Year',
        y='Forest Cover',
        color='Country Name',
        title=f"Forest Cover Over Time (Top 5 Countries from {selected_year.replace('Forest Area ', '')})",
        labels={'Forest Cover': 'Forest Cover %'},
        height=400
    )
    fig_time.update_traces(
        hovertemplate='Year: %{x}<br>Forest Cover: %{y:.2f}%'
    )
    
    return fig_map, fig_top, fig_bottom, fig_time

if __name__ == '__main__': 
    app.run(debug=True, port = 8053)