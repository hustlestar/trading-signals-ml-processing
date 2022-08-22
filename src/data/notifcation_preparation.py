import json

from dotenv import dotenv_values

from config import get_connection
from data.db import execute_sql


def flat_notifications_from_sql(notifications):
    return [{
        "id": r[0],
        "ticker": r[7],
        "price": float(r[6]),
        "notification_date": r[5],
        "highest_since_notified": float(r[3] if r[3] else 0),
        "lowest_since_notified": float(r[4] if r[4] else 0),
        "basis": json.loads(r[1]),
        "filter_name": r[2],
        "same_signal_in_last_3_hour": r[10],
        "same_signal_in_last_24_hours": r[11],
        "same_signal_profit_in_last_3_hours": r[12],
        "same_signal_profit_in_last_24_hours": r[13],
    } for r in notifications]


def flat_notifications_from_json(notifications):
    return [{
        "id": r["id"],
        "ticker": r["ticker"],
        "price": float(r["price"]),
        "notification_date": r["notification_date"] if r.get("notification_date") else r.get("notificationDate"),
        "highest_since_notified":
            float(r["highest_since_notified"] if r.get("highest_since_notified") else float(r["highestSinceNotified"]) if r.get("highestSinceNotified") else 0),
        "lowest_since_notified":
            float(r["lowest_since_notified"] if r.get("lowest_since_notified") else float(r["lowestSinceNotified"]) if r.get("lowestSinceNotified") else 0),
        "basis": json.loads(r["basis"]),
        "filter_name": r["filter_name"] if r.get('filter_name') else r.get("filterName")
    } for r in notifications]


def flatten_dict(d, prefix=None):
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flattened = flatten_dict(v, f"{k}_")
            for i, j in flattened.items():
                res[i if not prefix else f"{prefix}{i.lstrip('_')}"] = j
        else:
            res[k if not prefix else f"{prefix}{k.lstrip('_')}"] = v
    return res


def flatten_bars(bars_list, prefix=None, number_of_bars=60, reverse=False):
    res = {}
    for i, r in zip(range(number_of_bars), reversed(bars_list) if reverse else bars_list):
        for k, v in flatten_dict(r, f"{prefix if prefix else '_'}{process_digits(i)}_").items():
            res[k] = v
    return res


def process_digits(i):
    return i + 1 if i >= 9 else f"0{i + 1}"


def shrink_minutes(minutely):
    low = 9999999
    high = 0
    vol = 0
    for k, v in minutely.items():
        if k.endswith('_high') and v > high:
            high = v
        if k.endswith('_low') and v < low:
            low = v
        if k.endswith('_volume') and v > 0:
            vol = vol + v
    return {
        'latest_hour_open': minutely['current_min_bars_01_open'],
        'latest_hour_high': high,
        'latest_hour_low': low,
        'latest_hour_close': minutely[[k for k in minutely.keys() if k.endswith('_close')][-1]],
        'latest_hour_volume': vol
    }


def prepare_dataset(notifications):
    res = []
    for n in notifications:
        basis = n['basis']
        history = basis['history']
        btc_stats = basis.get('btcStats')
        history_flat = flatten_dict(history, "history_")
        btc_stats_flat = flatten_dict(btc_stats, "btc_stats_") if btc_stats else {}
        current = basis['current']
        current_flat = flatten_dict(current, "current_")
        minutely = flatten_bars(current_flat["current_minutelyBars"], "current_min_bars_", reverse=True)
        hourly = flatten_bars(current_flat["current_hourlyBars"], prefix="current_hour_bars_", number_of_bars=48)
        minutes_as_hour = shrink_minutes(minutely)
        res.append({
            **{k: v for k, v in n.items() if k not in ['basis']},
            **history_flat,
            **{k: v for k, v in current_flat.items() if k not in {"current_minutelyBars", "current_hourlyBars"}},
            **minutely,
            **minutes_as_hour,
            **hourly,
            **btc_stats_flat,
            "same_signal_in_last_3_hour": n["same_signal_in_last_3_hour"] if n.get("same_signal_in_last_3_hour") else 0,
            "same_signal_in_last_24_hour": n["same_signal_in_last_24_hour"] if n.get("same_signal_in_last_24_hour") else 0,
            "same_signal_profit_in_last_3_hours": n["same_signal_profit_in_last_3_hours"] if n.get("same_signal_profit_in_last_3_hours") else 0,
            "same_signal_profit_in_last_24_hours": n["same_signal_profit_in_last_24_hours"] if n.get("same_signal_profit_in_last_24_hours") else 0,
        })
    return res


if __name__ == '__main__':
    conf = dotenv_values("../../.env")
    conn = get_connection(conf)
    notifications = execute_sql(conn, """select
                                   *
                                   ,(select count(*) from notifications as q where 
                                     extract(epoch from n.notification_date - q.notification_date) / 60  between 0 and 180
                                     and q.notification_date < n.notification_date
                                     and q.ticker = n.ticker 
                                     and q.filter_name = n.filter_name
                                   ) as same_signal_in_last_3_hours
                                   ,(select count(*) from notifications as q where 
                                     extract(epoch from n.notification_date - q.notification_date) / 60  between 0 and 1440
                                     and q.notification_date < n.notification_date
                                     and q.ticker = n.ticker 
                                     and q.filter_name = n.filter_name
                                   ) as same_signal_in_last_24_hours
                                   ,(select TRUNC(sum((n.price - q.price)/ q.price * 100), 2)  from notifications as q where 
                                     extract(epoch from n.notification_date - q.notification_date) / 60  between 0 and 180
                                     and q.notification_date < n.notification_date
                                     and q.ticker = n.ticker 
                                     and q.filter_name = n.filter_name
                                   ) as same_signal_profit_in_last_3_hours
                                   ,(select TRUNC(sum((n.price - q.price)/ q.price * 100), 2)  from notifications as q where 
                                     extract(epoch from n.notification_date - q.notification_date) / 60  between 0 and 1440
                                     and q.notification_date < n.notification_date
                                     and q.ticker = n.ticker 
                                     and q.filter_name = n.filter_name
                                   ) as same_signal_profit_in_last_24_hours
                                   from notifications n
                                   order by id
                                   limit 500;""")
    notifications_list = flat_notifications_from_sql(notifications)

    res = prepare_dataset(notifications_list)
