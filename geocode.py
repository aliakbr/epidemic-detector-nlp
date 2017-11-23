import pandas as pd
import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
#%%

cred = credentials.Certificate("praecantatio-f846b-firebase-adminsdk-cj3ro-b847906b6b.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://praecantatio-f846b.firebaseio.com/'})
#%%

df = pd.read_pickle('tweets_to_classify.pkl')
df['class'] = pd.read_pickle('result.pkl')['class']

#%%

dfres = df[df['class'] == 1].copy()
dfres['coords_pair'] = pd.Series()

dfresplace = dfres[dfres.coordinates.isna() & dfres.place.notna()].copy()
dfrescoords = dfres[dfres.coordinates.notna()].copy()

#%%

def calc_coords(place):
    lat, lon = 0, 0
    count = 0
    if place != place or not place or not place['bounding_box'] or 'coordinates' not in place['bounding_box']:
        return None
    for lt, ln in place['bounding_box']['coordinates'][0]:
        count += 1
        lat += lt
        lon += ln

    return lat/count, lon/count

#%%

dfresplace['coords_pair'] = dfresplace['place'].apply(calc_coords)
dfrescoords['coords_pair'] = dfrescoords['coordinates'].apply(lambda x: float('nan') if x != x else tuple(x['coordinates']))

#%%

dffinal = pd.concat([dfresplace, dfrescoords])

#%%

ref = db.reference('flumap')
for lon, lat in dffinal.coords_pair:
    ref.push({
        'lat': lat,
        'lng': lon
    })

# with open('data.json', 'w') as wf:
#     json.dump(list(dffinal['coords_pair']), wf)
