import json
import pandas as pd
import re

def getActorsRating(actors):
	#import list of famous actors and create a list out of it
	actorRank_raw = pd.read_csv('data/actors.csv', usecols=['Name'], encoding='latin-1')
	actorRank = list(actorRank_raw['Name'])

	r = 0 #actor rating
	actornf = 0 #count of actors who were not in the list of famous actors(for debug)

	for a in actors: #iterate through each actor
		if a in actorRank: #if actor is in the top 1000
			r += int((actorRank.index(a)+10)/100) #add its index to r
		else: #if actor is not in list 
			actornf += 1 #increment 'actor not found' variable(for debug)

	return r

def add_actor_ratings(data):
	zerocount = 0
	 #import cast and crew information for movies
	credit = pd.read_csv('data/credits.csv')
	credit['id'] = credit['id'].apply(pd.to_numeric)
	tmplist = [] #container to store the actors per movie
	rating = 0 #container to store actor rating per movie
	count = 0 #number of entries that cannot be converted to json(for debug)
	emp = 0 #number of entries with no actor information(for debug)
	credit = credit.dropna()
	 #create new dataset with movie id and actor rating
	actorRatings = pd.DataFrame(columns = ['actor_rating','id'])
	print("Begin Evaluating Actor Ratings for movies:")
	for i in range(0,len(credit["cast"])): #go through each ith entry

		if i % 1000 == 0:
			print("Movie ", i)

		tmplist.clear()  #list to store the actors in movie to get actor index

		j=credit["cast"][i]
		j=j.encode('ascii', 'ignore').decode('ascii','ignore')  #eliminates special characters

		regex = '(".+?\'.+?")|(\'".+?"\')'

		j=re.sub(regex,'\'?\'',j) #this csv file does a wierd thing when there's an apostrophe in the name, so delete those

		j=j.replace("'", "\"") #convert ' into " for json to work
		j=j.replace("None", "0")  #convert 'profile_path': None into  'profile_pahth': 0; json doesn't work otherwise

		try: #using 'try' because some data is not suited for json
			obj = json.loads(j)

			for entry in obj:
				tmplist.append(entry["name"]) #make a list of actors

			if tmplist == []:
				emp += 1 #increment ' # of movies with no actors' count(debug)

			rating = getActorsRating(tmplist) #get actor rating

		except: #bad entry from csv file

			rating = 0
			count += 1  #increment bad entry count(debug)

		#add actor rating and id of this movie to dataset
		actorRatings = actorRatings.append(pd.DataFrame([[rating, credit["id"][i]]], columns = ['actor_rating','id']), ignore_index=True )

	return data.join(actorRatings.drop(["id"], axis=1)).fillna(value=0)
