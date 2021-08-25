import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
data = pd.read_csv("train.csv")
# data = data.query("type == 'conventional' and region == 'Albany'")
# data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
# data.sort_values("Date", inplace=True)
data.head()
app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="test",),
        html.P(
            children="test python dashbroad render",
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": data["date"],
                        "y": data["sales"],
                        "type": "lines",
                    },
                ],
                "layout": {"title": "date & sales"},
            },
        )
    ]


)

if __name__ == "__main__":
    app.run_server(debug=True)