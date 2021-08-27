import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



df = pd.read_csv("train.csv",
        parse_dates=['date'], 
        index_col="date")

df=df[df['item']==3]

#add features 
weekend=[]
holiday=[]
sg_holidays = holidays.SG()
for i, row in df.iterrows():
    weekno=i.weekday()
    
    if weekno < 5:
        weekend.append(0)
    else:  # 5 Sat, 6 Sun
        weekend.append(1)
        
    if i in sg_holidays:
        holiday.append(1)
    else:
        holiday.append(0)

df['is_weekend']=weekend
df['is_holiday']=holiday
df=df.drop(['store','item'],axis=1)

#train test split
train_size = int(len(df) * 0.9)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
# print(len(train), len(test))


#encode data
f_columns = ['is_weekend']
# f_columns = ['is_weekend', 'is_holiday']


f_transformer = RobustScaler()
sales_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
sales_transformer = sales_transformer.fit(train[['sales']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['sales'] = sales_transformer.transform(train[['sales']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['sales'] = sales_transformer.transform(test[['sales']])


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train, train.sales, time_steps)
X_test, y_test = create_dataset(test, test.sales, time_steps)



model = keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=128, 
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
)
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')



history = model.fit(
    X_train, y_train, 
    epochs=5, 
    batch_size=32, 
    validation_split=0.1,
    verbose=1,
    shuffle=False
)

train_pred = model.predict(X_train)
valid_pred = model.predict(X_test)
print("LSTM:")
print('Train rmse:', np.sqrt(mean_squared_error(y_train, train_pred)))
print('Validation rmse:', np.sqrt(mean_squared_error(y_test, valid_pred)))


#dashbroad
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
                        "x": df.index,
                        "y": df["sales"],
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