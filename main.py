from modules.database import init_db, load_usernames_from_csv


# create database and load usernames from csv
init_db()

# load usernames from csv
load_usernames_from_csv()