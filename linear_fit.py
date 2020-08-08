from sklearn import linear_model
from dataframe import data
from sklearn.feature_selection import RFECV
data = data.drop(["month", "popularity"], axis=1)

def linear_feature_select(feature_name, poly_columns, exclude = []):
	regr = linear_model.LinearRegression()

	x = data.drop([feature_name] + exclude, axis=1)
	for column in [e for e in poly_columns if e in x.columns]:
		x[column + "^2"] = list(map(lambda x : pow(x,2), x[column]))
		x[column + "^3"] = list(map(lambda x : pow(x,3), x[column]))
		x[column + "^4"] = list(map(lambda x : pow(x,4), x[column]))

	y = data[feature_name]

	selector = RFECV(regr, cv=5)

	fit = selector.fit(x, y)

	selected_cols = []

	for i in range(0, len(x.columns)):
		if fit.support_[i]:
			selected_cols.append(x.columns[i])

	print(feature_name + " fit selected features: " + str(selected_cols))

def linear_fit(feature_name, poly_columns,  training_percentage, exclude = []):
	regr = linear_model.LinearRegression()
	x = data.drop([feature_name] + exclude, axis=1)
	for column in [e for e in poly_columns if e in x.columns]:
		x[column + " squared"] = list(map(lambda x : pow(x,2), x[column]))
		x[column + " cubed"] = list(map(lambda x : pow(x,3), x[column]))

	y = data[feature_name]

	x_training = x[:int(len(x)*training_percentage)]
	x_testing =  x[int(len(x)*training_percentage):]
	y_training = y[:int(len(y)*training_percentage)]
	y_testing =  y[int(len(y)*training_percentage):]

	selector = RFECV(regr, cv=5)

	fit = selector.fit(x_training, y_training)

	print(feature_name + " score: " + str(fit.score(x_testing, y_testing)))

def elastic_linear_fit(feature_name, poly_columns,  training_percentage, exclude = []):
	l1_ratio = [.1, .3, .5, .7, .9, .95, .99, 1]
	regr = linear_model.ElasticNetCV(
		l1_ratio=l1_ratio,
		max_iter=4000,
		normalize=True,
		cv=10)

	x = data.drop([feature_name] + exclude, axis=1)
	for column in [e for e in poly_columns if e in x.columns]:
		x[column + "^2"] = list(map(lambda x : pow(x,2), x[column]))
		x[column + "^3"] = list(map(lambda x : pow(x,3), x[column]))
		x[column + "^4"] = list(map(lambda x : pow(x,4), x[column]))

	y = data[feature_name]

	x_training = x[:int(len(x)*training_percentage)]
	x_testing =  x[int(len(x)*training_percentage):]
	y_training = y[:int(len(y)*training_percentage)]
	y_testing =  y[int(len(y)*training_percentage):]

	regr.fit(x_training, y_training)
	# very small alpha for vote_average -> convergence warning
	print("alpha: " + str(regr.alpha_))
	print("coefficients: " + str(regr.coef_))
	print("l1 ratio: " + str(regr.l1_ratio_))

	print(feature_name + " score: " + str(regr.score(x_testing, y_testing)))
