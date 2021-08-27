import os
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement, BatchStatement

dir_path = os.path.dirname(os.path.realpath(__file__))
BUNDLE_PATH = os.path.join(dir_path, 'secure-connect-gesture-prediction.zip')

cloud_config = {
    'secure_connect_bundle': BUNDLE_PATH
}
auth_provider = PlainTextAuthProvider('UzJkZZyFwGSAxJCcKGLXtoEu',
                                      'ZntHG+Xa01.deSCmpuEGt1fRY_G+ptJgZZR9SluCpAk9Zz2Cqtkg2v7Q4FwpXi3-1qD7I9UMQzsNZsONamSju6f9fRGy4D8xWRJZJjvvGDYYaXiv1wmZxkqUk6W4bNhM')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()


def delete_data():
    query = f"""
            TRUNCATE gesture_prediction.gesture_prediction
            """
    session.execute(query)


def add_data(raw_df):
    delete_data()
    df = raw_df.copy()
    col = """id, time, channel1, channel2, channel3,
          channel4, channel5, channel6, channel7, channel8,
          class"""
    # for _, row in df.iterrows():
    #     query = f"""
    #     INSERT INTO gesture_prediction.gesture_prediction ({col})
    #     VALUES (uuid(), {int(row['time'])},{row['channel1']},{row['channel2']},
    #             {row['channel3']},{row['channel4']},{row['channel5']},
    #             {row['channel6']},{row['channel7']},{row['channel8']},{int(row['class'])})
    #     """
    #     session.execute(query)

    batch = BatchStatement()

    for _, row in df.iterrows():
        INSERT_STMT = f"""
            INSERT INTO gesture_prediction.gesture_prediction ({col})
            VALUES (uuid(), {int(row['time'])},{row['channel1']},{row['channel2']},
                    {row['channel3']},{row['channel4']},{row['channel5']},
                    {row['channel6']},{row['channel7']},{row['channel8']},{int(row['class'])})
            """
        batch.add(SimpleStatement(INSERT_STMT))

    session.execute(batch)
