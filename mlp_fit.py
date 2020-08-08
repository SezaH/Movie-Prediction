from mlp_model import mlp_model

best_layer_count = 2
best_node_count = 6

# classify_transform = function to apply to y to get it to be a boolean

def mlp_optimize(x, y, training_percentage):
	x_training = x[:int(len(x)*training_percentage)]
	x_testing =  x[int(len(x)*training_percentage):]
	y_training = y[:int(len(y)*training_percentage)]
	y_testing =  y[int(len(y)*training_percentage):]

	for i in range(1, 6):
		for j in range(4, 13, 3):
			score = mlp_fit(x, y, i, j, training_percentage)

			print(i, "layers", j, "nodes per layer: ", score)

def mlp_fit(x, y, num_hidden_layers, nodes_per_layer, training_percentage):
	x_training = x[:int(len(x)*training_percentage)]
	x_testing =  x[int(len(x)*training_percentage):]
	y_training = y[:int(len(y)*training_percentage)]
	y_testing =  y[int(len(y)*training_percentage):]

	mlp = mlp_model(num_hidden_layers, nodes_per_layer)
	mlp.fit(x_training, y_training)

	return mlp.score(x_testing, y_testing)
