import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
#%%

cred = credentials.Certificate("praecantatio-f846b-firebase-adminsdk-cj3ro-b847906b6b.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://praecantatio-f846b.firebaseio.com/'})

#%%

ref = db.reference('flumap')

ref.push({
    'lat': -6.912,
    'lng': 107.6097,
    'username': 'dumagummy',
    'text': 'I\'m sick with flu'
})
