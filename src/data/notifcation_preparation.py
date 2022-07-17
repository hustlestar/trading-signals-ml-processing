import json


def flat_notifications_from_sql(notifications):
    return [{
        "id": r[0],
        "ticker": r[7],
        "price": float(r[6]),
        "notification_date": r[5],
        "highest_since_notified": float(r[3] if r[3] else 0),
        "lowest_since_notified": float(r[4] if r[4] else 0),
        "basis": json.loads(r[1]),
        "filter_name": r[2]
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
        res.append({
            **{k: v for k, v in n.items() if k not in ['basis']},
            **history_flat,
            **{k: v for k, v in current_flat.items() if k not in {"current_minutelyBars", "current_hourlyBars"}},
            **minutely,
            **hourly,
            **btc_stats_flat
        })
    return res
