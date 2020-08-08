from sklearn.neural_network import MLPClassifier

def mlp_model(layerNumber, nodeNumber):
	layer_tuple = ()

	for i in range(0,layerNumber):
		layer_tuple += (nodeNumber,)

	return MLPClassifier(solver="adam", activation="relu", batch_size=50, n_iter_no_change=20, learning_rate_init=0.01, hidden_layer_sizes=layer_tuple, max_iter=1000)
