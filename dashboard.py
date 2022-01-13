import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import pandas as pd

from NBA_official_visualization import *




#################################################################################################################################
#                                                                                                                               #
#                                                       CREATE GRAPHS BLOCKS                                                    #
#                                                                                                                               #
#################################################################################################################################
df = pd.read_csv('datasets/df_players_merged.csv').drop(["Unnamed: 0"], axis=1)
df_match = pd.read_csv('datasets/df_matchs.csv').drop(["Unnamed: 0"], axis=1)

fig_general_1 = taille_distribution(df)
fig_general_2 = map_plot(df)
fig_pca_1 = pca_plot(df)
fig_poste_1 = plot_taille_poids(df)
fig_poste_2 = plot_type_paniers(df)
fig_pts_1 = pts2(df)
fig_pts_2 = pts3(df)
fig_match_1 = plot_match(df_match)


graph_gen_1 = dcc.Graph(
            id='gen1',
            figure=fig_general_1
            )
graph_gen_2 = dcc.Graph(
            id='gen2',
            figure=fig_general_2
            )
graph_pca_1 = dcc.Graph(
            id='pca2',
            figure=fig_pca_1
            )
graph_poste_1 = dcc.Graph(
            id='poste1',
            figure=fig_poste_1
            )
graph_poste_2 = dcc.Graph(
            id='poste2',
            figure=fig_poste_2
            )
graph_pts_1 = dcc.Graph(
            id='pts1',
            figure=fig_pts_1
            )
graph_pts_2 = dcc.Graph(
            id='pts2',
            figure=fig_pts_2
            )
graph_match_1 = dcc.Graph(
            id='match1',
            figure=fig_match_1
            )


players_comp_drop1 = dcc.Dropdown(
                id='p1',
                options=[{'label':i,'value':i} for i in df['PLAYER'].unique()],
                value='Kevin Durant'
            )


players_comp_drop2 = dcc.Dropdown(
                id='p2',
                options=[{'label':i,'value':i} for i in df['PLAYER'].unique()],
                value='Stephen Curry'
            )
players_comp_graph = dcc.Graph(
            id='player-comparaison',
            )


#################################################################################################################################
#                                                                                                                               #
#                                                               MAIN                                                            #
#                                                                                                                               #
#################################################################################################################################
app = dash.Dash(__name__)
app.title = "NBA data analysis"
app.layout = html.Div([   
        html.H1('NBA Data Analysis',style={'textAlign': 'center'}),
        
        # Line 1
        html.Br(),
        html.Br(),
        html.H2('Global overview'),
        html.Div([
            # Col 1
            graph_gen_1,
            # Col 2
            graph_gen_2,
        ], style={'display': 'flex', 'flex-direction': 'row'}),
        
        # Line 2
        html.Br(),
        html.Br(),
        html.H2('Positions analysis'),
        html.Div([
            # Col 1
            graph_poste_1,
            # Col 2
            graph_poste_2,
        ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row'}),

        
        # Line 3
        html.Br(),
        html.Br(),
        html.H2('Players comparison'),
        html.Div([
            # Col 1
            html.Div([html.Label('Player 1'),
                      players_comp_drop1,
                      html.Br(),
                      html.Label('Player 2'),
                      players_comp_drop2,
                      players_comp_graph
                      ], style={'padding': 10, 'flex': 1}),
            # Col 2
            html.Div(
                [html.H3("Glossaire :"),
                dcc.Markdown('**GP :** Game Played'),
                dcc.Markdown('**3PM :** 3 Point Field Goals Made'),
                dcc.Markdown('**FGM :** Field Goal Made'),
                dcc.Markdown('**REB :** Rebounds'),
                dcc.Markdown('**AST :** Assists'),
                dcc.Markdown('**STL :** Steals'),
                dcc.Markdown('**TOV :** Turnovers'),
            ], style={'padding': 10, 'flex': 1}),
            

        ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row'}),

        # Line 4
        html.Br(),
        html.Br(),
        html.H2("Analysis of the players' accuracy"),
        html.Div([
            # Col 1
            graph_pts_1,
            # Col 2
            graph_pts_2
        ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row'}),

        # Line 5
        html.Br(),
        html.Br(),
        html.H2("Teams win rate "),
        graph_match_1,
        
])




@app.callback(
    Output('player-comparaison', 'figure'),
    [Input('p1', 'value'),Input('p2', 'value')])
def players_comparaison(p1, p2):
    """Radar plot des caractéristiques des joueurs"""
    players = [i for i in [p1,p2] if i != None]
    caracs = ['GP',"3PM","FGM",'REB','AST','STL','TOV']
    df_plot = df[caracs+['PLAYER']]
    df_hover = df[caracs+['PLAYER']]
    for c in caracs:
        df_plot[c] = df_plot[c]/df_plot[c].max()

    colors = ["#542583","#fea500"]
    fillcolors = ["rgba(84, 37, 131, 0.5)","rgba(254, 165, 0, 0.5)"]
    fig = go.Figure()
    for i,p in enumerate(players):
        fig.add_trace(go.Scatterpolar(
                r=df_plot[df_plot['PLAYER'] == p].iloc[0,:-1],
                theta=caracs,
                fill='toself',
                line=dict(color=colors[i]),
                fillcolor=fillcolors[i],
                name=p,
                hoverinfo="text",
                hovertext=df_hover[df_hover['PLAYER'] == p].iloc[0,:-1],
        ))
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=True,
    )
    fig.for_each_trace(lambda t: t.update(hoveron='points'))
    
    return fig



if __name__ == '__main__':
    app.run_server(debug=True)














    
