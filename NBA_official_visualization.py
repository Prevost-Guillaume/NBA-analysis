import time
import pandas as pd
import numpy as np
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


    
#################################################################################################################################
#                                                                                                                               #
#                                                     ANALYSE DES POSTES                                                        #
#                                                                                                                               #
#################################################################################################################################
def plot_taille_poids(df):
    """Affiche le pose en fct de la taille et du poids des joueurs"""
    df = df.rename(columns={'PDV':'Position', 'Taille':'Height', 'Poids':'Weight'})
    fig = px.scatter(df, x='Height', y='Weight',
                     color='Position',
                     hover_name="PLAYER",
                     marginal_y = "box",
                     marginal_x = "box",
                     title='Position of the players according to their size',
                     color_discrete_sequence=['#000000','#6e518c','#fea500'])
    return fig




def plot_type_paniers(df, mode='bar'):
    """Nb de paniers 3pts et 2pts en fct du poste"""
    fig = go.Figure()

    df = df.rename(columns={'FG%':'2 points field goals', '3P%':'3 points field goals'})
    postes = ['G','F','C']
    colors = ['#542583','#000000','#fea500']
    caracs = ['2 points field goals','3 points field goals']
    #caracs = ['FG%','3P%']
    
    for i,p in enumerate(postes):
        df_plot = df[df['PDV'] == p]

        if mode == 'bar':
            x = [caracs[0], caracs[1]]
            y = [df_plot[caracs[0]].mean(), df_plot[caracs[1]].mean()]
        elif mode == 'box':
            x = [caracs[0] for _ in range(df_plot.shape[0])]+[caracs[1] for _ in range(df_plot.shape[0])]
            y = list(df_plot[caracs[0]])+list(df_plot[caracs[1]])
            
        fig.add_trace(go.Bar(
            y=y, x=x,
            name=p,
            marker_color=colors[i]))

    fig.update_layout(
        yaxis_title='Average number',
        boxmode='group',
        title='Average number of 2 and 3 point baskets by position'
    )
     
    return fig


#################################################################################################################################
#                                                                                                                               #
#                                                   ANALYSE GLOBALE JOUEUERS NBA                                                #
#                                                                                                                               #
#################################################################################################################################

def map_plot(df):
    df_count = df.groupby(['country','iso_alpha']).size()

    df_viz = pd.DataFrame()
    df_viz['country'] = [i[0] for i in df_count.index]
    df_viz['iso_alpha'] = [i[1] for i in df_count.index]
    df_viz['count'] = list(df_count.apply(lambda x:np.log(x)))
    df_viz['taille'] = list(df.groupby('iso_alpha')['Taille'].mean())

    fig = px.choropleth(df_viz, locations="iso_alpha",
                        color="count",
                        hover_name="country",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="Origin of the players (logarithmic scale)")
    return fig


def taille_distribution(df):
    fig = px.histogram(df['Taille'], nbins=50)
    #fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)

    mean_human = 178.4
    std_human = 7.59

    x_pdf = np.linspace(140, 240, 200)
    y_pdf = scipy.stats.norm.pdf(x_pdf, mean_human, std_human)

    x_pdf2 = np.linspace(140, 240, 200)
    y_pdf2 = scipy.stats.norm.pdf(x_pdf, (df['Taille']*100).mean(), (df['Taille']*100).std())

    # Plotly
    fig = go.Figure()
##    fig.add_trace(
##        go.Histogram(
##            x=df['Taille']*100,
##            histnorm='probability density',
##            nbinsx=15,
##            name='Taille des joueurs de la NBA',
##        ))
    fig.add_trace(
        go.Scatter(
            x=x_pdf, y=y_pdf,
            line=dict(color="#542583"),
            fill='tozeroy',
            fillcolor='rgba(84, 37, 131, 0.5)',
            name="Men's height",
        ))
    fig.add_trace(
        go.Scatter(
            x=x_pdf2,y=y_pdf2,
            line=dict(color="#fea500"),
            fill='tozeroy',
            fillcolor = 'rgba(254, 165, 0, 0.5)',
            name='Height of NBA players',
        ))
    fig.update_layout(
        title='Height comparison between men and NBA players',
        xaxis_title='Height (cm)',
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    
    return fig
    


def players_comparaison(df, players=[]):
    """Radar plot des caractéristiques des joueurs"""
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
      title='Player comparison'
    )
    fig.for_each_trace(lambda t: t.update(hoveron='points'))
    
    return fig



def pts2(df):
    attempted = df['FGA']
    scored = df['FGM']

    # make the linear regression
    X = attempted.values.reshape(-1,1)
    y = scored
    regression = LinearRegression()
    regression.fit(X,y)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = regression.predict(x_range.reshape(-1, 1))

    # identifying best shooters
    df['best_worst'] = (scored-attempted*regression.coef_)
    df=df.sort_values(by=['best_worst'])

    df['best_worst'][:5]=-100 #set the five worst to -1 -> in blue on the graphic
    df['best_worst'][-5:]=100 #set the five best to 1 -> in yellow on the graphic
    df['best_worst'][5:-5]=0 #set the other to 0

    df = df.rename(columns={'FGA': 'Field Goals Attempted Per Match', 'FGM': 'Field Goals Scored Per Match'})
    # plot the graph
    fig = px.scatter(df, x='Field Goals Attempted Per Match', y='Field Goals Scored Per Match', hover_name='PLAYER',title=f"Average accuracy of 2-point baskets : {round(regression.coef_[0],2)}",template="plotly_white",color='best_worst')
    fig.add_traces(go.Scatter(x=x_range, y=y_range, name=' '))
    fig.update_layout(showlegend=False)
    fig.update_coloraxes(showscale=False)
    return fig



def pts3(df):
    attempted = df['3PA']
    scored = df['3PM']

    # make the linear regression
    X = attempted.values.reshape(-1,1)
    y = scored
    regression = LinearRegression()
    regression.fit(X,y)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = regression.predict(x_range.reshape(-1, 1))

    # identifying best shooters
    df['best_worst'] = (scored-attempted*regression.coef_)
    df=df.sort_values(by=['best_worst'])

    df['best_worst'][:5]=-100 #set the five worst to -1
    df['best_worst'][-5:]=100 #set the five best to 1
    df['best_worst'][5:-5]=0 #set the other to 0

    df = df.rename(columns={'3PA': '3 Points Shots Attempted Per Match', '3PM': '3 Points Shots Scored Per Match'})
    # plot the graph
    fig = px.scatter(df,x='3 Points Shots Attempted Per Match', y='3 Points Shots Scored Per Match', hover_name='PLAYER',title=f"Average accuracy of 3-point baskets : {round(regression.coef_[0],2)}",template="plotly_white",color='best_worst')
    fig.add_traces(go.Scatter(x=x_range, y=y_range, name=' '))
    fig.update_layout(showlegend=False)
    fig.update_coloraxes(showscale=False)
    return fig



    
#################################################################################################################################
#                                                                                                                               #
#                                                               PCA                                                             #
#                                                                                                                               #
#################################################################################################################################
def pca_plot(df):
    pca = make_pipeline(StandardScaler(), PCA(n_components=2))
    
    x = df[["FG%","3PM","FGM","+/-","AST%","OREB%","DREB%","REB%","%FGA\n2PT","%FGA\n3PT","%PTS","%REB","%TOV","%STL","%BLK","%PF","3FGM\n%UAST"]]

    x = list(pca.fit_transform(np.array(x)))
    df['rank'] = df.index
    fig = px.scatter(df, x=[i[0] for i in x], y=[i[1] for i in x],
                     color="rank",
                     hover_name="PLAYER",
                     title="Players' statistics")

    #fig.layout.paper_bgcolor = '#FFFFFF'
    #fig.layout.plot_bgcolor = '#FFFFFF'
    return fig

def pca_plot_teams(df):
    pca = make_pipeline(StandardScaler(), PCA(n_components=2))
    
    x = df[[i for i in df.columns if i not in ['TEAM', 'GP', 'W', 'L']]]

    x = list(pca.fit_transform(np.array(x)))
    df['rank'] = df.index
    fig = px.scatter(df, x=[i[0] for i in x], y=[i[1] for i in x],
                     color="rank",
                     hover_name="TEAM",
                     title="Teams' statistics")

    #fig.layout.paper_bgcolor = '#FFFFFF'
    #fig.layout.plot_bgcolor = '#FFFFFF'
    return fig


#################################################################################################################################
#                                                                                                                               #
#                                                               MATCHS                                                          #
#                                                                                                                               #
#################################################################################################################################
def plot_match(df):
    dates = 2009+df.index
    fig = go.Figure()
    for i in df.columns:
        if i not in ['Durant', 'Wilbon', 'Stephen A', 'Basketball Club of Brazil', 'Giannis', 'Home', 'Away', '36ers', 'Breakers', 'Wildcats', 'Ducks','Stephen', 'LeBron', 'Bullets', 'United', 'Sharks', 'Long-Lions', 'TBD', 'San Lorenzo', 'FC Barcelona Lassa', 'Canada', 'Paschoalotto/Bauru', 'Olimpia Milano', 'Fenerbahce Sports Club', 'USA', 'World', 'Flamengo', 'Maccabi Electra', 'Webber', 'Hill', 'Basket', 'Chuck', 'Shaq', 'FC Barcelona Regal', 'EA7 Emporio Armani Milano', 'Montepaschi Siena', 'Alba Berlin', 'Fenerbahce Ulker', 'Team Chuck', 'Team Shaq', 'Dallas', 'Armani Jeans Milano', 'Maccabi Haifa', 'Regal FC', 'CSKA', 'Caja Laboral', 'Maccabi Elite', 'Olympiacos', 'Real Madrid','Partizan', 'West', 'East', 'Sophomores', 'Rookies']  :
            x = dates
            y = df[i]

            fig.add_trace(go.Scatter(
                y=y, x=x,
                #opacity=0.1 if i not in [0,25,14,18] else 1,
                name=i,
                visible=None if i in ('Nets', 'Rockets') else 'legendonly'))

            fig.update_layout(
                yaxis_title='% wins',
                xaxis_title='year',
                title='Evolution of the percentage of wins over time',
                legend_title_text='Teams',

            )

    return fig


#################################################################################################################################
#                                                                                                                               #
#                                                               MAIN                                                            #
#                                                                                                                               #
#################################################################################################################################

if __name__ == '__main__':

    df = pd.read_csv('datasets/df_players_merged.csv').drop(["Unnamed: 0"], axis=1)


    fig = pca_plot(df)
    fig.show()
    fig = plot_taille_poids(df)
    fig.show()
    fig = map_plot(df)
    fig.show()
    fig = plot_type_paniers(df)
    fig.show()
    fig = taille_distribution(df)
    fig.show()
    fig = pts2(df)
    fig.show()
    fig = pts3(df)
    fig.show()    
    fig = players_comparaison(df, players=['Kevin Durant', 'Stephen Curry'])
    fig.show()

    df_teams = pd.read_csv('datasets/df_teams_merged.csv').drop(["Unnamed: 0"], axis=1)
    fig = pca_plot_teams(df_teams)
    fig.show()

    df_match = pd.read_csv('datasets/df_matchs.csv').drop(["Unnamed: 0"], axis=1)
    fig = plot_match(df_match)
    fig.show()
    




    








    
