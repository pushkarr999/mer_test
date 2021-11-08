from flask import Flask, request, jsonify
import utils

app = Flask(__name__)


@app.route('/hello')
def hello():
    return "hi"


@app.route('/get_price', methods=['POST'])
def predict_price():
    ##post and get json request
    data = request.get_json()
    if not (data.get('name') is None):
        name = data.get('name')
    else:
        name = data.get('name', 'unk_name')
    if not (data.get('item_condition_id') is None):
        item_condition_id = data.get('item_condition_id')
    else:
        item_condition_id = data.get('item_condition_id', 2)  # set to mode of train.csv
    if not (data.get('category_name') is None):
        category_name = data.get('category_name')
    else:
        category_name = data.get('category_name', 'unk_cat')
    if not (data.get('brand_name') is None):
        brand_name = data.get('brand_name')
    else:
        category_name = data.get('brand_name', 'unk_brand')
    if not (data.get('item_description') is None):
        item_description = data.get('item_description')
    else:
        item_description = data.get('item_description', 'unk_descr')
    if not (data.get('shipping') is None):
        shipping = data.get('shipping')
    else:
        shipping = data.get('shipping', 0)
    if not (data.get('seller_if') is None):
        seller_id = data.get('seller_id')
    else:
        seller_id = data.get('seller_id', 1841452510)

    response = jsonify({
        'price': utils.get_prediction(name, item_condition_id, category_name, brand_name, shipping, seller_id,
                                      item_description)
    }
    )
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server")
    app.run(host="0.0.0.0", port=5000)
