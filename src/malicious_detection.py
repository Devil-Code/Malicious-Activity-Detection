# Author Pritesh Gandhi

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

kdd_train = pd.read_csv("./kdd.csv")

kdd_train.head()
kdd_train.shape

"""# data prepocessing"""

kdd_train.columns = ['train_duration', 'train_protocol_type', 'train_service', 'train_flag', 'src_bytes',
                     'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                     'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                     'num_shells', 'num_access_files', 'dummy', 'num_outbound_cmds', 'is_host_login',
                     'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                     'srv_rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                     'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                     'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                     'dst_host_srv_serror_rate', 'train_dst_host_rerror_rate', 'dst_bushost_srv_rerror_rate']

kdd_train.head()

# Dropping Unessasry coloumns
kdd_train_clean = kdd_train.drop(
    ['wrong_fragment', 'urgent', 'num_failed_logins', 'num_file_creations', 'num_shells', 'dummy',
     'num_outbound_cmds'], axis=1)

kdd_train_clean.head()

kdd_train_clean.info()

kdd_train_clean.describe()

train_protocol_type = {'tcp': 0, 'udp': 1, 'icmp': 2}
kdd_train_clean['train_protocol_type'].value_counts()

kdd_train_clean['train_service'].value_counts()

kdd_train_clean['train_flag'].value_counts()

kdd_train_clean['train_dst_host_rerror_rate'].value_counts()

train_protocol_type = {'tcp': 0, 'udp': 1, 'icmp': 2}
train_protocol_type.items()
kdd_train_clean.train_protocol_type = [train_protocol_type[item] for item in kdd_train_clean.train_protocol_type]
kdd_train_clean.head(20)

train_duration = kdd_train_clean['train_duration']

kdd_train_clean['train_duration'] = np.where((kdd_train_clean.train_duration <= 2), 0, 1)

kdd_train_clean.head(20)

train_replace_map = {'normal': 'normal', 'DOS': ['back', 'land', 'pod', 'neptune', 'smurf', 'teardrop'],
                     'R2L': ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'spy', 'phf', 'warezclient',
                             'warezmaster'], 'U2R': ['buffer_overflow', 'loadmodule', 'perl', 'rootkit'],
                     'PROBE': ['ipsweep', 'nmap', 'portsweep', 'satan']}

kdd_train_format = kdd_train_clean.assign(
    train_dst_host_rerror_rate=kdd_train_clean['train_dst_host_rerror_rate'].apply(
        lambda x: [key for key, value in train_replace_map.items() if str(x) in value]))

kdd_train_format.head(20)

train_service = {'aol': 1, 'auth': 2, 'bgp': 3, 'courier': 4, 'csnet_ns': 5, 'ctf': 6, 'daytime': 7, 'discard': 8,
                 'domain': 9, 'domain_u': 10, 'echo': 11, 'eco_i': 12, 'ecr_i': 13, 'efs': 14, 'exec': 15,
                 'finger': 16, 'ftp': 17, 'ftp_data': 18, 'gopher': 19, 'harvest': 20, 'hostnames': 21, 'http': 22,
                 'http_2784': 23, 'http_443': 24, 'http_8001': 25, 'imap4': 26, 'IRC': 27, 'iso_tsap': 28,
                 'klogin': 29, 'kshell': 30, 'ldap': 31, 'link': 32, 'login': 33, 'mtp': 34, 'name': 35,
                 'netbios_dgm': 36, 'netbios_ns': 37, 'netbios_ssn': 38, 'netstat': 39, 'nnsp': 40, 'nntp': 41,
                 'ntp_u': 42, 'other': 43, 'pm_dump': 44, 'pop_2': 45, 'pop_3': 46, 'printer': 47, 'private': 48,
                 'red_i': 49, 'remote_job': 50, 'rje': 51, 'shell': 52, 'smtp': 53, 'sql_net': 54, 'ssh': 55,
                 'sunrpc': 56, 'supdup': 57, 'systat': 58, 'telnet': 59, 'tftp_u': 60, 'tim_i': 61, 'time': 62,
                 'urh_i': 63, 'urp_i': 64, 'uucp': 65, 'uucp_path': 66, 'vmnet': 67, 'whois': 68, 'X11': 69,
                 'Z39_50': 70}

train_service.items()

kdd_train_format.train_service = [train_service[item] for item in kdd_train_format.train_service]

kdd_train_format.head(20)

train_dst_host_rerror_rate = {'normal': 0, 'DOS': 1, 'R2L': 2, 'U2R': 3, 'PROBE': 4}

train_dst_host_rerror_rate.items()

kdd_train_format.train_dst_host_rerror_rate = [train_dst_host_rerror_rate[item[0]] for item in kdd_train_format.train_dst_host_rerror_rate]

train_flag = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'S1': 5, 'SH': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9,
              'OTH': 10}

train_flag.items()

kdd_train_format.train_flag = [train_flag[item] for item in kdd_train_format.train_flag]

kdd_train_format.head(20)

y = kdd_train_format.iloc[:, -2]
y

x = kdd_train_format.drop(['train_dst_host_rerror_rate'], axis=1)
x

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=100)

#RANDOM FORESTS:
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

rf.score(X_train,y_train),rf.score(X_test,y_test)

y_pred_test = rf.predict(X_test)

print(classification_report(y_test, y_pred_test))

confusion_matrix(y_test, y_pred_test)

