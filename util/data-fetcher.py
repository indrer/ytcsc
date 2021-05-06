import pandas as pd
import psycopg2
from secret import database_connect
from secret import query

column_names = ['body', 'positive', 'negative', 'neutral', 'rated', 'comment_id', 'video_id', 'date']

try:
    connection = psycopg2.connect(**database_connect)
    cur = connection.cursor()
    cur.execute(query)
    data = cur.fetchall()
    cur.close()
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv('../datasets/labelled/unpreprocessed.csv', index=False)
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if connection is not None:
        connection.close()
        print('Database connection closed.')

