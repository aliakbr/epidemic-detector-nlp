import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("praecantatio-f846b-firebase-adminsdk-cj3ro-b847906b6b.json")
firebase_admin.initialize_app(cred)
