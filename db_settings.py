import sqlite3


def connect_to_db():
    conn = sqlite3.connect('weatherbot.db')
    return conn

def update_table(data, conn):

    cursor = conn.cursor()

    create_table_query = """CREATE TABLE IF NOT EXISTS weather_history (
            username VARCHAR(50),
            date DATE,
            location VARCHAR(50),
            min_temp VARCHAR(50),
            max_temp VARCHAR(50),
            avg_temp VARCHAR(50),
            pressure VARCHAR(50)
        )"""

    cursor.execute(create_table_query)

    username = data['username']
    selected_date = data['selected_date']
    place = data['place']
    min_temp_pred = data['min_temp_pred']
    max_temp_pred = data['max_temp_pred']
    avg_temp_pred = data['avg_temp_pred']
    pressure_pred = data['pressure_pred']

    insert_query = f'''
        INSERT INTO weather_history (username, date, location, min_temp, max_temp, avg_temp, pressure)
        VALUES ('{username}', '{selected_date}', '{place}', '{min_temp_pred}', '{max_temp_pred}', '{avg_temp_pred}', '{pressure_pred}')
    '''
    cursor.execute(insert_query)
    conn.commit()

    print("Done adding new data for:", place)


def get_all_data_from_db(username, conn):

    cursor = conn.cursor()

    select_query = f'''
        SELECT *
        FROM weather_history
        WHERE username = '{username}'
    '''

    cursor.execute(select_query)
    result = cursor.fetchall()

    cursor.close()
    conn.close()

    return result

# if __name__ == "__main__":
#     data = {
#         'username': "mike",
#         'place': "Maiduguri",
#         'selected_date': '23/05/2023',
#         'min_temp_pred': '20',
#         'max_temp_pred': '32',
#         'avg_temp_pred': '26',
#         'pressure_pred': '10000',
#         'current_date': '23/05/2023'
#     }
#     conn = connect_to_db()
#     # update_table(data, conn)
#     print(get_all_data_from_db('mike', conn))