import logging


def execute_sql(conn, query):
    logging.info('Executing SQL Query & Fetching Results...')
    db_cursor = conn.cursor()
    db_cursor.execute(query)
    records = db_cursor.fetchall()
    logging.info(f'Finished SQL Query Execution, returning {len(records)} rows')
    return records


def batched_read_notification_sql(conn, batch_size=1000, subset=0):
    logging.info(f"Executing batched read of notifications with batch size = {batch_size} and subset = {subset}")
    notifications = []
    max_id = execute_sql(conn, "select max(id) as max_id from notifications")
    max_id = max_id[0][0]

    full_range = range(0, max_id, batch_size)
    if subset:
        full_range = full_range[:subset]
    for i in full_range:
        start = i
        end = i + batch_size
        query = f"""select
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
        where id >= {start} and id < {end} 
        order by id
        """
        logging.info(f"Running batch id >= {start} and id < {end}")
        print(f"Running batch id >= {start} and id < {end}")
        notifications = notifications + execute_sql(conn, query)

    return notifications
