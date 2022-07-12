import requests
import logging

logging.basicConfig(level=logging.ERROR)

from config import get_connection
from data.notifcation_preparation import prepare_dataset, flat_notifications_from_json
from data.db import execute_sql
import json

conn = get_connection()
# notifications = execute_sql(conn, "SELECT json_agg(k) FROM (select * from notifications where id = 61113) k")


for i in range(63000, 64000):
    notifications = execute_sql(conn, f"SELECT json_agg(k) FROM (select * from notifications where id = {i}) k")
    if notifications:
        try:
            notifications_raw = notifications[0][0][0]
            response = requests.post('http://localhost:5000/predict', json=notifications_raw)
            print(json.dumps(notifications_raw))
            if '1' in response.content.decode('utf-8'):
                print(
                    f"res for {i}: {response} : {response.content if '1' not in response.content.decode('utf-8') else response.content.decode('utf-8') + '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'}")
        except:
            pass

# raw_data = prepare_dataset(flat_notifications_from_json([notifications_raw]))

# print(raw_data)
