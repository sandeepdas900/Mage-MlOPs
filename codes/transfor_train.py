from typing import Tuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd

# Assuming you have a decorator for transformers, adjust the name accordingly.
if 'transformer' not in globals():
    # Import the transformer decorator from your specific library or package.
    from mage_ai.data_preparation.decorators import transformer

@transformer
def train_model(df: pd.DataFrame) -> Tuple[DictVectorizer, LinearRegression]:
    # Clean and prepare the dataset
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    categorical = ['PULocationID', 'DOLocationID']  # Separate features for pickup and dropoff
    df[categorical] = df[categorical].astype(str)
    
    # Change to dictionary and extract target variable
    train_dicts = df[categorical].to_dict(orient='records')
    y_train = df['duration'].values
    
    # Initialize DictVectorizer and Linear Regression model
    dv = DictVectorizer()
    lr = LinearRegression()
    
    # Fit DictVectorizer and transform features
    X_train = dv.fit_transform(train_dicts)
    
    # Train the Linear Regression model
    lr.fit(X_train, y_train)
    
    # Return both the DictVectorizer and the trained model
    return dv, lr

# We will now read the January data from the link
df_jan = pd.read_parquet('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

# Now let's train our model with the dataset we've read
dv, lr = train_model(df_jan)

# To print the intercept of the trained model
intercept = lr.intercept_
print(f'The intercept of the model is: {intercept}')
