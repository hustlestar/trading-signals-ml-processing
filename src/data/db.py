import logging
from typing import List


def execute_sql(conn, query):
    logging.info('Executing SQL Query & Fetching Results...')
    db_cursor = conn.cursor()
    db_cursor.execute(query)
    records = db_cursor.fetchall()
    logging.info(f'Finished SQL Query Execution, returning {len(records)} rows')
    return records


def batched_read_notification_sql(conn, batch_size=1000, subset=0) -> List:
    logging.info(f"Executing batched read of notifications with batch size = {batch_size} and subset = {subset}")
    notifications = []
    max_id = execute_sql(conn, "select max(id) as max_id from notifications")
    max_id = max_id[0][0]

    full_range = range(0, max_id, batch_size)
    if subset:
        full_range = full_range[:subset]
    # for i in full_range:
    #     start = i
    #     end = i + batch_size
    query = f"""with data_over_3 as (
select *,
    -- deduct because we don't want current row included
    (count(*) over w) - 1 as same_event_in_prior_3_hours,
    (sum(1 / price) over w) - 1 / price as reciprocal_sum_3
from notifications
window w as (partition by ticker, filter_name
             order by notification_date
             range between interval '3 hours' preceding and current row)
),
data_over_24 as (
select *,
    -- deduct because we don't want current row included
    (count(*) over w) - 1 as same_event_in_prior_24_hours,
    (sum(1 / price) over w) - 1 / price as reciprocal_sum_24
from notifications
window w as (partition by ticker, filter_name
             order by notification_date
             range between interval '3 hours' preceding and current row)
)
select o3.*, o24.same_event_in_prior_24_hours, o24.reciprocal_sum_24,
	TRUNC(100 * (o3.price * o3.reciprocal_sum_3 - o3.same_event_in_prior_3_hours), 2)
	as same_signal_profit_in_last_3_hours,
	TRUNC(100 * (o24.price * o24.reciprocal_sum_24 - o24.same_event_in_prior_24_hours), 2)
	as same_signal_profit_in_last_24_hours
from data_over_3 as o3
inner join data_over_24 as o24 on o3.id = o24.id
order by id"""
    # logging.info(f"Running batch id >= {start} and id < {end}")
    logging.info(f"Running batch id MEGA")
    print(f"Running batch id MEGA")
    notifications = notifications + execute_sql(conn, query)

    return notifications
