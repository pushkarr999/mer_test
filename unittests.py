import requests
from flask import jsonify

from server import app
import unittest

app.testing = True


class FlaskTestCase(unittest.TestCase):
    datajson = {
        "name": "coin necklac",
        "item_condition_id": 1,
        "category_name": 'Women/Jewelry/Necklaces',
        "shipping": 0,
        "brand_name": 'forever 21',
        "seller_id": 2982673593,
        "item_description": 'silver'
    }
    url = "http://127.0.0.1:5000/get_price"

    # Check if Response is 200
    def test_1_index(self):
        tester = app.test_client(self)
        response = tester.get("/hello")
        status_code = response.status_code
        self.assertEqual(status_code, 200)

    def test_2_get_price(self):
        tester = app.test_client(self)
        r = tester.post(FlaskTestCase.url, json=FlaskTestCase.datajson)
        self.assertEqual(r.status_code, 200)

    ## check if return json
    def test_3_return(self):
        tester = app.test_client(self)
        r = tester.post(FlaskTestCase.url, json=FlaskTestCase.datajson)
        self.assertEqual(r.content_type,"application/json")

    ## check if price is there
    def test_4_check(self):
        tester = app.test_client(self)
        r = tester.post(FlaskTestCase.url,json=FlaskTestCase.datajson)
        self.assertTrue(b'price' in r.data)


if __name__ == "__main__":
    unittest.main()
