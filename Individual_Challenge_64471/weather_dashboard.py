import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Load and Prepare the Data 
try:
    
    df = pd.read_csv('filtered data/GlobalWeatherRepository.csv')

    
    
    df.rename(columns={
        'Country': 'Country_Name',
        'location_name': 'Location', 
        'latitude': 'Latitude',      
        'longitude': 'Longitude',
        'last_updated': 'Date',
        'temperature_celsius': 'Temperature',
        'precip_mm': 'Precipitation',
        'humidity': 'Humidity',
        'air_quality_PM2.5': 'Air_Quality_PM2_5',
        'air_quality_PM10': 'Air_Quality_PM10'
    }, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    numeric_cols = ['Temperature', 'Humidity', 'Precipitation', 'Latitude', 'Longitude', 'Air_Quality_PM2_5', 'Air_Quality_PM10']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where essential numerical data or Date is missing
    df.dropna(subset=['Date', 'Temperature', 'Humidity', 'Precipitation', 'Latitude', 'Longitude'], inplace=True)

    location_summary_df = df.groupby(['Country_Name', 'Location', 'Latitude', 'Longitude']).agg(
        Average_Temperature=('Temperature', 'mean'),
        Average_Humidity=('Humidity', 'mean'),
        Total_Precipitation=('Precipitation', 'sum'),
        Average_Air_Quality_PM2_5=('Air_Quality_PM2_5', 'mean'),
        Average_Air_Quality_PM10=('Air_Quality_PM10', 'mean'),
        Count=('Date', 'count') 
    ).reset_index()

    country_summary_df = df.groupby(['Country_Name', 'Date']).agg(
        Average_Temperature=('Temperature', 'mean'),
        Average_Humidity=('Humidity', 'mean'),
        Total_Precipitation=('Precipitation', 'sum'),
        Average_Air_Quality_PM2_5=('Air_Quality_PM2_5', 'mean'),
        Average_Air_Quality_PM10=('Air_Quality_PM10', 'mean')
    ).reset_index()

    overall_country_summary_df = df.groupby('Country_Name').agg(
        Average_Temperature=('Temperature', 'mean'),
        Average_Humidity=('Humidity', 'mean'),
        Total_Precipitation=('Precipitation', 'sum'),
        Average_Air_Quality_PM2_5=('Air_Quality_PM2_5', 'mean'),
        Average_Air_Quality_PM10=('Air_Quality_PM10', 'mean')
    ).reset_index()


    unique_countries = sorted(df['Country_Name'].unique())

    data_load_error = False
except Exception as e:
    print(f"Error loading or processing data: {e}")
    df = pd.DataFrame() 
    location_summary_df = pd.DataFrame()
    country_summary_df = pd.DataFrame()
    overall_country_summary_df = pd.DataFrame() # Initialize the DataFrame
    unique_countries = []
    data_load_error = True

# Create the Dash App 
app = dash.Dash(__name__)
server = app.server 

# Define the App Layout
app.layout = html.Div([
    html.H1("Global Weather Dashboard", style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#2d3748'}),

    html.Div(id='data-error-message', style={'color': 'red', 'textAlign': 'center', 'fontSize': '1.2em', 'marginBottom': '20px'}),

    html.Div([
        
        html.Div([
            html.Label("Select Metric:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#4a5568'}),
            dcc.Dropdown(
                id='metric-selector',
                options=[
                    {'label': 'Average Temperature (°C)', 'value': 'Average_Temperature'},
                    {'label': 'Average Humidity (%)', 'value': 'Average_Humidity'},
                    {'label': 'Total Precipitation (mm)', 'value': 'Total_Precipitation'},
                    {'label': 'Average Air Quality (PM2.5)', 'value': 'Average_Air_Quality_PM2_5'},
                    {'label': 'Average Air Quality (PM10)', 'value': 'Average_Air_Quality_PM10'}
                ],
                value='Average_Temperature', 
                clearable=False,
                style={'width': '100%', 'borderRadius': '0.5rem', 'border': '1px solid #cbd5e0'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}), 

        # Country Selector Dropdown for Time Series
        html.Div([
            html.Label("Select Country (for Time Series):", style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#4a5568'}),
            dcc.Dropdown(
                id='country-selector',
                options=[{'label': country, 'value': country} for country in unique_countries],
                value=unique_countries[0] if unique_countries else None, 
                placeholder="Select a country...",
                clearable=True, 
                style={'width': '100%', 'borderRadius': '0.5rem', 'border': '1px solid #cbd5e0'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}), 

    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px', 'backgroundColor': '#ffffff', 'padding': '1rem', 'borderRadius': '0.75rem', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),

    html.Div([
        
        html.Div([
            dcc.Graph(
                id='top-locations', 
                style={'height': '400px', 'borderRadius': '0.75rem', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            dcc.Graph(
                id='bottom-locations', 
                style={'height': '400px', 'borderRadius': '0.75rem', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'float': 'right'}),
    ], style={'width': '100%', 'marginBottom': '20px', 'display': 'flex', 'justifyContent': 'space-between'}),

    # Time Series Plot
    html.Div([
        dcc.Graph(
            id='time-series',
            style={'height': '400px', 'borderRadius': '0.75rem', 'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'}
        )
    ], style={'width': '100%', 'marginBottom': '20px'}),

    html.Div([
        html.P("Data Source: Global Weather Repository", style={'textAlign': 'center', 'fontSize': 12, 'color': '#718096'})
    ])
], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#f0f4f8', 'borderRadius': '0.75rem'})

# Define Callbacks
@app.callback(
    [Output('data-error-message', 'children'),
     Output('top-locations', 'figure'),
     Output('bottom-locations', 'figure'),
     Output('time-series', 'figure')
    ],
    [Input('metric-selector', 'value'),
     Input('country-selector', 'value')]
)
def update_figures(selected_metric, selected_country):
    
    if data_load_error or df.empty or location_summary_df.empty or country_summary_df.empty or overall_country_summary_df.empty:
        error_msg = "Error: Could not load or process 'GlobalWeatherRepository.csv'. Please ensure the file exists and is correctly formatted, and contains sufficient data."
        empty_figure = go.Figure()
        
        return error_msg, empty_figure, empty_figure, empty_figure

    metric_labels = {
        'Average_Temperature': {'title': 'Average Temperature', 'unit': '°C', 'colorscale': 'Plasma'},
        'Average_Humidity': {'title': 'Average Humidity', 'unit': '%', 'colorscale': 'Blues'},
        'Total_Precipitation': {'title': 'Total Precipitation', 'unit': 'mm', 'colorscale': 'Viridis'},
        'Average_Air_Quality_PM2_5': {'title': 'Average Air Quality (PM2.5)', 'unit': 'µg/m³', 'colorscale': 'Hot'},
        'Average_Air_Quality_PM10': {'title': 'Average Air Quality (PM10)', 'unit': 'µg/m³', 'colorscale': 'Hot'}
    }
    current_metric_info = metric_labels.get(selected_metric, {'title': '', 'unit': '', 'colorscale': 'Viridis'})
    y_axis_label = f"{current_metric_info['title']} ({current_metric_info['unit']})"
    color_scale = current_metric_info['colorscale']

    sorted_countries = overall_country_summary_df.sort_values(by=selected_metric, ascending=False)

    # Top Countries Bar Chart
    fig_top = go.Figure() 
    if not sorted_countries.empty and selected_metric in sorted_countries.columns:
        top_countries = sorted_countries.head(10)
        if not top_countries.empty:
            fig_top = px.bar(
                top_countries,
                x='Country_Name', 
                y=selected_metric,
                title=f"Top 10 Countries by {current_metric_info['title']}", 
                color=selected_metric,
                color_continuous_scale=color_scale,
                labels={selected_metric: y_axis_label},
                height=400
            )
            fig_top.update_traces(
                hovertemplate=f'%{{x}}<br>%{{y:.2f}} {current_metric_info["unit"]}'
            )
            fig_top.update_layout(xaxis_title="Country", yaxis_title=y_axis_label) 
        else:
            fig_top = go.Figure().add_annotation(text="No data for top countries.",
                                                 xref="paper", yref="paper",
                                                 x=0.5, y=0.5, showarrow=False,
                                                 font=dict(size=16, color="gray"))
    else:
        fig_top = go.Figure().add_annotation(text="No data for top countries.",
                                             xref="paper", yref="paper",
                                             x=0.5, y=0.5, showarrow=False,
                                             font=dict(size=16, color="gray"))


    # Bottom Countries Bar Chart
    fig_bottom = go.Figure() 
    if not sorted_countries.empty and selected_metric in sorted_countries.columns:
        bottom_countries = sorted_countries.tail(10).sort_values(by=selected_metric, ascending=True)
        if not bottom_countries.empty:
            fig_bottom = px.bar(
                bottom_countries,
                x='Country_Name',
                y=selected_metric,
                title=f"Bottom 10 Countries by {current_metric_info['title']}", 
                color=selected_metric,
                color_continuous_scale=color_scale,
                labels={selected_metric: y_axis_label},
                height=400
            )
            fig_bottom.update_traces(
                hovertemplate=f'%{{x}}<br>%{{y:.2f}} {current_metric_info["unit"]}'
            )
            fig_bottom.update_layout(xaxis_title="Country", yaxis_title=y_axis_label) 
        else:
            fig_bottom = go.Figure().add_annotation(text="No data for bottom countries.",
                                                    xref="paper", yref="paper",
                                                    x=0.5, y=0.5, showarrow=False,
                                                    font=dict(size=16, color="gray"))
    else:
        fig_bottom = go.Figure().add_annotation(text="No data for bottom countries.",
                                                xref="paper", yref="paper",
                                                x=0.5, y=0.5, showarrow=False,
                                                font=dict(size=16, color="gray"))


    # Time Series Plot (by Country)
    fig_time = go.Figure() 
    time_series_data_available = False

    if selected_country:
        
        time_series_df_filtered = country_summary_df[country_summary_df['Country_Name'] == selected_country].sort_values(by='Date')
        plot_title = f"{current_metric_info['title']} Over Time in {selected_country}"

        if not time_series_df_filtered.empty and selected_metric in time_series_df_filtered.columns:
            time_series_data_available = True
            time_series_plot_data = time_series_df_filtered
    else:
        
        global_avg_time_series_df = country_summary_df.groupby('Date').agg(
            Average_Temperature=('Average_Temperature', 'mean'),
            Average_Humidity=('Average_Humidity', 'mean'),
            Total_Precipitation=('Total_Precipitation', 'sum'), 
            Average_Air_Quality_PM2_5=('Average_Air_Quality_PM2_5', 'mean'),
            Average_Air_Quality_PM10=('Average_Air_Quality_PM10', 'mean')
        ).reset_index().sort_values(by='Date')
        plot_title = f"Global Average {current_metric_info['title']} Over Time"

        if not global_avg_time_series_df.empty and selected_metric in global_avg_time_series_df.columns:
            time_series_data_available = True
            time_series_plot_data = global_avg_time_series_df
        else:
            time_series_plot_data = pd.DataFrame() 

    if time_series_data_available:
        fig_time = px.line(
            time_series_plot_data, 
            x='Date',
            y=selected_metric,
            title=plot_title,
            labels={'Date': 'Date', selected_metric: y_axis_label},
            height=400
        )
        fig_time.update_traces(
            mode='lines+markers',
            marker=dict(size=5),
            hovertemplate=f'Date: %{{x|%Y-%m-%d}}<br>{current_metric_info["title"]}: %{{y:.2f}} {current_metric_info["unit"]}'
        )
        fig_time.update_layout(
            xaxis_title="Date",
            yaxis_title=y_axis_label,
            hovermode="x unified"
        )
    else:
        fig_time = go.Figure().add_annotation(text="No time series data available for this selection.",
                                              xref="paper", yref="paper",
                                              x=0.5, y=0.5, showarrow=False,
                                              font=dict(size=16, color="gray"))

    return "", fig_top, fig_bottom, fig_time

# Run the App 
if __name__ == '__main__':
   
    if data_load_error:
        print("Dashboard cannot start due to data loading errors. Please check 'GlobalWeatherRepository.csv'.")
    else:
        app.run(debug=True, port=5057)
