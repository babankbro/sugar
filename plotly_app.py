import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import json

app = dash.Dash(__name__)
df = pd.DataFrame({
   'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 'Bananas'],
   'Amount': [4, 1, 2, 2, 4, 5],
   'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
})
fig = px.bar(df, x='Fruit', y='Amount', color='City',  
   barmode='group')

with open('result.json') as file:
  df2 = json.load(file)

df = [dict(Task="Job A", Subtask="Subtask 1", Start='2020-02-01', Finish='2020-05-02'),
      dict(Task="Job A", Subtask="Subtask 2", Start='2020-03-01', Finish='2020-07-02'),
      dict(Task="Job A", Subtask="Subtask 3", Start='2020-04-01', Finish='2020-10-02'),
      dict(Task="Job B", Subtask="Subtask 1", Start='2020-03-01', Finish='2020-06-02'),
      dict(Task="Job B", Subtask="Subtask 2", Start='2020-05-01', Finish='2020-08-02'),
      dict(Task="Job B", Subtask="Subtask 3", Start='2020-06-01', Finish='2020-09-02'),
      dict(Task="Job C", Subtask="Subtask 1", Start='2020-01-01', Finish='2020-04-02'),
      dict(Task="Job C", Subtask="Subtask 2", Start='2020-05-01', Finish='2020-08-02'),
      dict(Task="Job C", Subtask="Subtask 3", Start='2020-06-01', Finish='2020-09-02')]

fig = ff.create_gantt(df2, colors=['#FF4040', '#FFFF40', '#40FF40', '#0022FF' ,  '#888AAA',  '#BBBBBB'], index_col='Subtask',
                      show_colorbar=True, bar_width=0.2, showgrid_x=True, showgrid_y=True, group_tasks=True)

#fig.show()

app.layout = html.Div(children=[
   html.H1(children='Hello Dash'),
   html.Div(children='''
   Dash: A web application framework for Python.
   '''),
   dcc.Graph(
      id='example-graph',
      figure=fig
   )
]) 
if __name__ == '__main__':
   app.run("0.0.0.0", 5000)
