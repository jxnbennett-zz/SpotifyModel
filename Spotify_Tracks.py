import sys
import spotipy
import spotipy.util as util
import numpy as np
import pandas as pd

# Authoriation Token (Authorization Code Flow)
token = util.prompt_for_user_token('username', 'scope',
                                   'client_id', 'client_secret',
                                   'https://example.com/callback/')
sp_play_mod = spotipy.Spotify(auth=token)

# Names of playlists to be used in model building
rel_playlists = ['Training 1', 'Training 2', 'Training 3', 'Training 4']

# Create a dictionary of all user playlists containing their IDs and total tracks
playlist_info = sp_play_mod.current_user_playlists()
play_names = dict()

# Nested dictionary where each playlist is the key and the items are its ID and total number of tracks
for PlayList in playlist_info['items']:
    play_names[PlayList['name']] = {'id': PlayList['id'], 'total': PlayList['tracks']['total']}

for my_list in rel_playlists:
    # Request limit is 50; Determine how many requests should be made
    upper = int((play_names[my_list]['total'] - play_names[my_list]['total'] % 50) / 50)

    index_list = list(range(0, upper + 1))

    for number in index_list:
        # Initialize lists for relevant variables
        track_ids = list()
        track_names = list()
        track_pop = list()
        track_expl = list()
        artist_ids = list()
        artist_names = list()
        genres = list()

        lim = 50

        results = sp_play_mod.user_playlist_tracks("username", play_names[my_list]['id'],
                                                   limit=lim, offset=number * 50)
        for item in results['items']:
            if item['track']['artists'][0]['id'] is None:
                continue
            track_ids.append(item['track']['id'])
            track_names.append(item['track']['name'])
            track_pop.append(item['track']['popularity'])
            track_expl.append(item['track']['explicit'])
            artist_ids.append(item['track']['artists'][0]['id'])
            artist_names.append(item['track']['artists'][0]['name'])

        artist_info = sp_play_mod.artists(artist_ids)
        for artist in artist_info['artists']:
            try:
                genres.append(artist['genres'][0])
            except:
                genres.append("Na")

        song_info = {'id': track_ids, 'artist_id': artist_ids, 'artist_name': artist_names, 'track_name': track_names,
                     'genre': genres, 'popularity': track_pop, 'explicit': track_expl}
        song_info_df = pd.DataFrame.from_dict(song_info)

        song_attributes = sp_play_mod.audio_features(track_ids)

        desired_atts = ["id", "danceability", "energy", "key", "loudness", "mode",
                        "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
                        "tempo", "duration_ms", "time_signature"]

        pre_att_df = dict()
        for att in desired_atts:
            att_list = list()
            for song in song_attributes:
                try:
                    att_list.append(song[att])
                except:
                    att_list.append("Na")
            pre_att_df[att] = att_list
        att_df = pd.DataFrame.from_dict(pre_att_df)

        full_df = pd.merge(left=song_info_df, right=att_df, on="id")

        if number == 0:
            complete_df = full_df
        else:
            complete_df = pd.concat([complete_df, full_df], ignore_index=True)
    title = my_list + ".csv"
    complete_df.to_csv(title, sep=",", encoding='utf=8')




