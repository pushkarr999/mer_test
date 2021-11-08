# Mercari Skill Test
Developing a price predition model and a REST endpoint.\
As the prices are in real numbers, this is a regression problem.
## Model Building
### Pre-processing
fill_missing_values - fills the missing brand names of the mercari_train.csv with "unk_brand".

Text data is the most unstructured form of data for machine learning models. Hence, it is processed and formatted properly.
Preprocessing will perform decontracting words, removing stop words, removing special characters and then apply stemming on the words in the sentence. 

Later missing brand names are guessed and "unk_brand" is replaced with existing list of brands compared with the name.

The category name is splitted in three categories for better prediction.\
A new feature is_expensive is added to indicate brands and products are that expensive in general.\
All the categorical features are one-hot encoded using TfidVectorizer.
Final sparse matrix is combined and used for training and testing the model.

preprocess.py contains all the preprocessing functions for new test data.
utils.py forms a feature pipeline and gets the final prediction.

### Training

The train-test split of 80%-20% is done on the train.csv for training the model.\
Model is trained using ridge regression testing with various value of learning rate and finally optimal alpha is found out.\
This is just an L2 regularized linear regression. This performs fairly well as compared to Lasso or unregularized regrassion. An L2 regularization never reduces the feature weights to zero. But it will surely reduce the weights for less important features.
This model gives fairly low RMSLE training error.

### Testing

Model is evaluated and mean squared log error is calculated. The model is exported as pickle to use for prediction of prices.
The prices for mercari_test.csv are exported in required format.

## API server
Runs on host="0.0.0.0", port=5000
### Endpoints
/get_price uses POST method to get the JSON input data and returns the predicted price in JSON format.

### Running tests
unittests.py contains the unit tests for the server.
After setting up the server, we can run the python file to perform all the unit tests.

### Building docker image
DockerFile is used to create a docker image that runs the API.
To create a docker image in server directory, run, 
```
docker build -f DockerFile -t price_pred_docker .
```

### Running docker image
After creating the docker image, we can see the image using,
```
docker image ls
```

To start the python flask server,
```
docker run -p 5000:5000 price_pred_docker
```
This will run the server on port 5000 and host 0.0.0.0.
We can perform post requests on http://0.0.0.0:5000/get_price




