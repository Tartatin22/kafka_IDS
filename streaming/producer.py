import json
import time
from datetime import datetime
from settings import TRANSACTIONS_TOPIC, DELAY, OUTLIERS_GENERATION_PROBABILITY
from streaming.utils import create_producer

_id = 0
producer = create_producer()

PATH = "data/KDDTest.csv"

columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
           "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
           "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
           "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
           "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
           "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
           "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "labels", "label"]

assert producer is not None

with open(PATH, "r") as file:
    _id = 0
    for line in file:
        d = dict(map(lambda i, j: (i, j), columns, line.strip().split(",")))
        d["id"] = _id
        _id += 1
        d["current_time"] = datetime.utcnow().isoformat()
        json_obj = json.dumps(d).encode("utf-8")

        producer.produce(topic=TRANSACTIONS_TOPIC,
                         value=json_obj)
        producer.flush()
        time.sleep(DELAY)

# sys.exit(0)
# breakpoint()


# if producer is not None:
#     while True:
#         # Generate some abnormal observations
#         if random.random() <= OUTLIERS_GENERATION_PROBABILITY:
#             X_test = np.random.uniform(low=-4, high=4, size=(1, 2))
#         else:
#             X = 0.3 * np.random.randn(1, 2)
#             X_test = (X + np.random.choice(a=[2, -2], size=1, p=[0.5, 0.5]))

#         X_test = np.round(X_test, 3).tolist()

#         current_time = datetime.utcnow().isoformat()

#         record = {"id": _id, "data": X_test, "current_time": current_time}
#         record = json.dumps(record).encode("utf-8")

#         producer.produce(topic=TRANSACTIONS_TOPIC,
#                          value=record)
#         producer.flush()
#         _id += 1
#         time.sleep(DELAY)
