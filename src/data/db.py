import logging


def execute_sql(conn, query):
    logging.info('Executing SQL Query & Fetching Results...')
    db_cursor = conn.cursor()
    db_cursor.execute(query)
    records = db_cursor.fetchall()
    logging.info(f'Finished SQL Query Execution, returning {len(records)} rows')
    return records