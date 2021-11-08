import pickle
import numpy as np
import pandas as pd

from scipy.sparse import hstack
import preprocess


def feature_pipeline(X_data, general_cat_vectorizer=None, subcat_1_vectorizer=None, subcat_2_vectorizer=None,
                     brand_name_vectorizer=None, item_name_vectorizer=None,
                     item_desc_vectorizer=None, scaler_name_len=None, scaler_desc_len=None):
    X_data = preprocess.fill_missing_values(X_data)
    X_data['item_description'] = X_data['item_description'].apply(
        preprocess.preprocess_descriptive_text_column)
    X_data['name'] = X_data['name'].apply(preprocess.preprocess_descriptive_text_column)
    X_data['brand_name'] = X_data['brand_name'].apply(lambda x: str(x).lower())
    X_data = preprocess.brand_guesser(X_data)
    name_len, scaler_name_len = preprocess.get_len_feature(X_data['name'], scaler_name_len)
    desc_len, scaler_desc_len = preprocess.get_len_feature(X_data['item_description'], scaler_desc_len)
    X_data = preprocess.split_categories(X_data)
    sparse_is_expensive = preprocess.get_is_expensive_feature(X_data)
    sparse_shipping = preprocess.get_shipping_feature(X_data)
    general_cat_ohe, general_cat_vectorizer = preprocess.vectorize_data(X_data['general_cat'].values.astype('U'),
                                                                        general_cat_vectorizer)
    subcat_1_ohe, subcat_1_vectorizer = preprocess.vectorize_data(X_data['subcat_1'].values.astype('U'),
                                                                  subcat_1_vectorizer)
    subcat_2_ohe, subcat_2_vectorizer = preprocess.vectorize_data(X_data['subcat_2'].values.astype('U'),
                                                                  subcat_2_vectorizer)
    brand_name_ohe, brand_name_vectorizer = preprocess.vectorize_data(X_data['brand_name'].values.astype('U'),
                                                                      brand_name_vectorizer)
    item_name_ohe, item_name_vectorizer = preprocess.vectorize_data(X_data['name'], item_name_vectorizer)
    item_desc_ohe, item_desc_vectorizer = preprocess.vectorize_data(X_data['item_description'], item_desc_vectorizer)
    X_featurized = hstack((general_cat_ohe, subcat_1_ohe, subcat_2_ohe, brand_name_ohe, item_name_ohe, item_desc_ohe,
                           desc_len, name_len, X_data['item_condition_id'].values.reshape(-1, 1),
                           sparse_shipping)).tocsr()
    return X_featurized, general_cat_vectorizer, subcat_1_vectorizer, subcat_2_vectorizer, brand_name_vectorizer, item_name_vectorizer, item_desc_vectorizer, scaler_name_len, scaler_desc_len


def get_prediction(name, item_condition_id, category_name, brand_name, shipping, seller_id, item_description):
    modeltest = pickle.load(open("./model/price_prediction_promising.pickle", "rb"))
    item_desc_vectorizer = pickle.load(open("./model/vectorizers/item_desc_vectorizer.pickle", "rb"))
    brand_name_vectorizer = pickle.load(open("./model/vectorizers/brand_name_vectorizer.pickle", "rb"))
    general_cat_vactorizer = pickle.load(open("./model/vectorizers/general_cat_vectorizer.pickle", "rb"))
    item_name_vectorizer = pickle.load(open("./model/vectorizers/item_name_vectorizer.pickle", "rb"))
    subcat_1_vectorizer = pickle.load(open("./model/vectorizers/subcat_1_vectorizer.pickle", "rb"))
    subcat_2_vectorizer = pickle.load(open("./model/vectorizers/subcat_2_vectorizer.pickle", "rb"))
    scaler_name_len = pickle.load(open("./model/vectorizers/scaler_name.pickle", "rb"))
    scaler_desc_len = pickle.load(open("./model/vectorizers/scaler_desc.pickle", "rb"))
    data = [[name, item_condition_id, category_name, brand_name, shipping, seller_id, item_description,
             category_name.split("/")[0], category_name.split("/")[1],
             category_name.split("/")[2]]]
    dftest = pd.DataFrame(data,
                          columns=['name', 'item_condition_id', 'category_name', 'brand_name', 'shipping', 'seller_id',
                                   'item_description', 'general_cat', 'subcat_1', 'subcat_2'])
    Xtetest, _, _, _, _, _, _, _, _ = feature_pipeline(dftest, general_cat_vactorizer, subcat_1_vectorizer,
                                                       subcat_2_vectorizer, brand_name_vectorizer,
                                                       item_name_vectorizer,
                                                       item_desc_vectorizer, scaler_name_len, scaler_desc_len)
    return np.expm1(modeltest.predict(Xtetest))[0]


if __name__ == '__main__':
    print(get_prediction('coin necklac', 1, 'Women/Jewelry/Necklaces', 'forever 21', 0, 2982673593, 'silver'))
