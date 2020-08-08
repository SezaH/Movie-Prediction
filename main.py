from linear_fit import linear_fit, linear_feature_select
from mlp_fit import mlp_fit, mlp_optimize
from dataframe import data, full_data
import numpy as np
from graph import graph
from scatter_matrix_2d import scatter_2d
import sys, os
from gui import MovieSuggestion
from data_stats import show_stats
from sklearn import preprocessing
import svd
import content_based_filtering as CB
from cf_metrics import cf_metrics
from cf_misc_stats import cf_misc_stats
from cluster_guess import cluster_guess

print("\n")

# part_to_run = which part of program to run. graphs, ANN, etc.
# ex: if you want to show only graphs, you would do "python main.py graphs"


if len(sys.argv) >= 2:
	part_to_run = sys.argv[1]
else:
	print("Please specify a part of the program to run.")
	exit()

if part_to_run in ["mlp", "mlp_optimize"]:
	training_percentage = 0.6

	data = data.drop(["month"], axis=1) # remove unhelpful features

	x = data.drop(["revenue", "vote_count", "vote_average", "popularity"], axis=1)
	x_no_budget = x.drop(["budget"], axis=1)
	x[["year", "budget", "runtime"]] = preprocessing.MinMaxScaler().fit_transform(x[["year", "budget", "runtime"]])

	if os.environ.get("WITH_ACTORS"):
		x[["actor_rating"]] = preprocessing.MinMaxScaler().fit_transform(x[["actor_rating"]])

	revenue_index = list(data.columns).index("revenue")
	budget_index  = list(data.columns).index("budget")

	profitable_y      = list(map(lambda r : int(r[revenue_index] - r[budget_index] > 0), data.values))
	popularity_y      = list(map(lambda x : int(x > np.median(data["popularity"])), data["popularity"]))
	vote_count_y      = list(map(lambda x : int(x >= np.average(data["vote_count"])), data["vote_count"]))
	vote_average_y    = list(map(lambda x : int(x >= 6), data["vote_average"]))
	budget_0_100k_y   = list(map(lambda x : int(x <= 100000), data["budget"]))
	budget_100k_1m_y  = list(map(lambda x : int(x > 100000 and x <= 1000000), data["budget"]))
	budget_1m_5m_y    = list(map(lambda x : int(x > 1000000 and x <= 5000000), data["budget"]))
	budget_5m_10m_y   = list(map(lambda x : int(x > 5000000 and x <= 10000000), data["budget"]))
	budget_10m_20m_y  = list(map(lambda x : int(x > 10000000 and x <= 20000000), data["budget"]))
	budget_20m_50m_y  = list(map(lambda x : int(x > 20000000 and x <= 50000000), data["budget"]))
	budget_50m_plus_y = list(map(lambda x : int(x > 50000000), data["budget"]))

	if part_to_run == "mlp":
		print("mlp:\n")

		print("number of profitable movies:", len(list(filter(lambda x : x == 1, profitable_y))))
		print("number of movies with budget between 0 and 100k:", len(list(filter(lambda x : x == 1, budget_0_100k_y))))
		print("number of movies with budget between 100k and 1m:", len(list(filter(lambda x : x == 1, budget_100k_1m_y))))
		print("number of movies with budget between 1m and 5m:", len(list(filter(lambda x : x == 1, budget_1m_5m_y))))
		print("number of movies with budget between 5m and 10m:", len(list(filter(lambda x : x == 1, budget_5m_10m_y))))
		print("number of movies with budget between 10m and 20m:", len(list(filter(lambda x : x == 1, budget_10m_20m_y))))
		print("number of movies with budget between 20m and 50m:", len(list(filter(lambda x : x == 1, budget_20m_50m_y))))
		print("number of movies with budget above 50m:", len(list(filter(lambda x : x == 1, budget_50m_plus_y))))

		# layer / node #s used based on optimization results
		profitable_score      = mlp_fit(x, profitable_y,                4, 7,  training_percentage)
		popularity_score      = mlp_fit(x, popularity_y,                1, 4,  training_percentage)
		vote_count_score      = mlp_fit(x, vote_count_y,                5, 10, training_percentage)
		vote_average_score    = mlp_fit(x, vote_average_y,              3, 10, training_percentage)
		budget_0_100k_score   = mlp_fit(x_no_budget, budget_0_100k_y,   3, 4,  training_percentage)
		budget_100k_1m_score  = mlp_fit(x_no_budget, budget_100k_1m_y,  3, 4,  training_percentage)
		budget_1m_5m_score    = mlp_fit(x_no_budget, budget_1m_5m_y,    3, 4,  training_percentage)
		budget_5m_10m_score   = mlp_fit(x_no_budget, budget_5m_10m_y,   3, 4,  training_percentage)
		budget_10m_20m_score  = mlp_fit(x_no_budget, budget_10m_20m_y,  3, 4,  training_percentage)
		budget_20m_50m_score  = mlp_fit(x_no_budget, budget_20m_50m_y,  3, 4,  training_percentage)
		budget_50m_plus_score = mlp_fit(x_no_budget, budget_50m_plus_y, 3, 4,  training_percentage)

		print("") # newline
		# numbers for num layers / nodes per layer are from param sweep
		print("revenue > budget score:",                         profitable_score)
		print("popularity > median score:",                      popularity_score)
		print("vote count >= average score:",                    vote_count_score)
		print("vote average >= 6 score:",                        vote_average_score)
		print("budget between 0 and 100,000 score:",             budget_0_100k_score)
		print("budget between 100,000 and 1,000,000 score:",     budget_100k_1m_score)
		print("budget between 1,000,000 and 5,000,000 score:",   budget_1m_5m_score)
		print("budget between 5,000,000 and 10,000,000 score:",  budget_5m_10m_score)
		print("budget between 10,000,000 and 20,000,000 score:", budget_10m_20m_score)
		print("budget between 20,000,000 and 50,000,000 score:", budget_20m_50m_score)
		print("budget above 50,000,000 score:",                  budget_50m_plus_score)
	else: # show optimization numbers
		print("mlp optimization:\n")

		print("revenue > budget:")
		mlp_optimize(x, profitable_y,                training_percentage)
		print("popularity > median:")
		mlp_optimize(x, popularity_y,                training_percentage)
		print("vote count >= average:")
		mlp_optimize(x, vote_count_y,                training_percentage)
		print("vote average >= 6:")
		mlp_optimize(x, vote_average_y,              training_percentage)
		print("budget between 0 and 100,000:")
		mlp_optimize(x_no_budget, budget_0_100k_y,   training_percentage)
		print("budget between 100,000 and 1,000,000:")
		mlp_optimize(x_no_budget, budget_100k_1m_y,  training_percentage)
		print("budget between 1,000,000 and 5,000,000:")
		mlp_optimize(x_no_budget, budget_1m_5m_y,    training_percentage)
		print("budget between 5,000,000 and 10,000,000:")
		mlp_optimize(x_no_budget, budget_5m_10m_y,   training_percentage)
		print("budget between 10,000,000 and 20,000,000:")
		mlp_optimize(x_no_budget, budget_10m_20m_y,  training_percentage)
		print("budget between 20,000,000 and 50,000,000:")
		mlp_optimize(x_no_budget, budget_20m_50m_y,  training_percentage)
		print("budget above 50,000,000:")
		mlp_optimize(x_no_budget, budget_50m_plus_y, training_percentage)

if part_to_run == "linreg":
	training_percentage = 0.7

	print("\nlinear:\n")

	poly_columns = ["vote_count", "budget", "revenue", "vote_average", "runtime"]

	linear_fit("vote_count", poly_columns, training_percentage, ["revenue", "vote_average"])
	linear_fit("budget", poly_columns, training_percentage, ["vote_count", "revenue", "vote_average"])
	linear_fit("revenue", poly_columns, training_percentage, ["vote_count", "vote_average"])
	linear_fit("vote_average", poly_columns, training_percentage, ["vote_count", "revenue"])

if part_to_run == "linreg_elastic":
	training_percentage = 0.7

	print("\nlinear (elastic):\n")

	poly_columns = ["vote_count", "budget", "revenue", "vote_average", "runtime"]

	linear_fit("vote_count", poly_columns, training_percentage)
	linear_fit("budget", poly_columns, training_percentage, ["vote_count", "revenue", "vote_average"])
	linear_fit("revenue", poly_columns, training_percentage, ["vote_count", "vote_average"])
	linear_fit("vote_average", poly_columns, training_percentage, ["vote_count", "revenue"])

if part_to_run == "linreg_feature_select":
	print("\nlinear feature select:\n")

	poly_columns = ["vote_count", "budget", "revenue", "vote_average", "runtime"]

	linear_feature_select("vote_count", poly_columns)
	linear_feature_select("budget", poly_columns, ["vote_count", "revenue", "vote_average"])
	linear_feature_select("revenue", poly_columns, ["vote_count", "vote_average"])
	linear_feature_select("vote_average", poly_columns, ["vote_count", "revenue"])

if part_to_run == "graphs":
	if len(sys.argv) != 4:
		print("usage: python main.py graphs feature1 feature2")

	graph(sys.argv[2], sys.argv[3])

if part_to_run == "2d_scatter":
	scatter_2d()

if part_to_run == "gui":
	MovieSuggestion().run()

if part_to_run == "stats":
	show_stats()

if part_to_run == "svd":
	svd.recommend(9)

if part_to_run == "cb":
	CB.get_recommendations(userId=9, n_recommendations=10)

if part_to_run == "cf_metrics":
	cf_metrics()

if part_to_run == "cf_misc_stats":
	cf_misc_stats()

if part_to_run == "cluster_guess":
	cluster_guess(data, False)

if part_to_run == "cluster_guess_plot":
	cluster_guess(data, True)
