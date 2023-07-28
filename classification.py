import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import copy
import os
import argparse

# get ../current working directory
prj_path = os.path.abspath(os.path.join(os.getcwd(), "../"))

# set random seed 
torch.manual_seed(0)

# #### Prepare Datasets
def prepare_datasets(shapes, train_percent, batch_size=64):
	features = []
	labels = []
	for shape in shapes:
		X_data = torch.from_numpy(np.load(open(f"{prj_path}/datasets/features_{shape}.npy", 'rb')))
		y_data = torch.from_numpy(np.load(open(f"{prj_path}/datasets/targets_{shape}.npy", 'rb')))
		X_data = X_data.to(torch.float32)
		y_data = y_data.to(torch.float32)
		features.append(X_data)
		labels.append(y_data)

	# convert to torch tensors
	features = torch.cat(features)
	labels = torch.cat(labels)

	# convert labels to three classes, 0 for negative, 1 for zero, 2 for positive
	for i in range(len(labels)):
		if labels[i] < 0:
			labels[i] = 0
		elif labels[i] == 0:
			labels[i] = 1
		else:
			labels[i] = 2

	# convert to int64
	labels = labels.to(torch.int64)

	# shuffle data
	perm = torch.randperm(len(features))
	features = features[perm]
	labels = labels[perm]

	# # Normalize the features
	# scaler = StandardScaler()
	# features = scaler.fit_transform(features.numpy())
	# features = torch.from_numpy(features)

	if train_percent == -1:
		dataset = TensorDataset(features, labels)
		return DataLoader(dataset, batch_size=batch_size)
	else:
		train_size = int(train_percent * len(features))

		train_dataset = TensorDataset(features[:train_size], labels[:train_size])
		test_dataset = TensorDataset(features[train_size:], labels[train_size:])

		train_dl = DataLoader(train_dataset, batch_size=batch_size)
		test_dl = DataLoader(test_dataset, batch_size=batch_size)
		return train_dl, test_dl

# #### Training and Testing Functions

def test(model, test_dl, device):
	correct = 0
	total = 0
	model.eval()
	with torch.no_grad():
		for X, y in test_dl:
			X = X.to(device)
			y = y.to(device)
			outputs = model(X)
			_, predicted = torch.max(outputs.data, 1)
			total += y.size(0)
			correct += (predicted == y).sum().item()
	accuracy = (correct / total)*100
	return accuracy

def train(model, optimizer, accuracy_fun, train_dl, device):
	model.train()
	for X, y in train_dl:
		X = X.to(device)
		y = y.to(device)
		optimizer.zero_grad()
		outputs = model(X)
		accuracy = accuracy_fun(outputs, y)
		accuracy.backward()
		optimizer.step()

def train_and_test(model, train_dl, test_dl, optimizer, accuracy_fun, device, num_epochs, plot_file, detailed_result_path):
	progress_bar = tqdm(range(num_epochs))
	best_weights = None
	best_accuracy = -np.inf
	# plot the train and test accuracy in each epoch
	train_accuracies = []
	test_accuracies = []
	with open(detailed_result_path, 'w') as f:
		print(f'epoch,train_accuracy,test_accuracy', file=f)
		for epoch in range(1, num_epochs + 1):
			train(model, optimizer, accuracy_fun, train_dl, device)
			train_accuracy = test(model, train_dl, device)
			test_accuracy = test(model, test_dl, device)
			train_accuracies.append(train_accuracy)
			test_accuracies.append(test_accuracy)
			print(f'{epoch:5d},{train_accuracy:10.3f},{test_accuracy:10.3f}', file=f)
			progress_bar.set_description(f'Epoch {epoch}, Test accuracy: {test_accuracy:.3f}%, Train accuracy: {train_accuracy:.3f}%')
			progress_bar.update(1)
			# save the best model weights
			if epoch == 1 or test_accuracy > best_accuracy:
				best_accuracy = test_accuracy
				# deep copy the model weights
				best_weights = copy.deepcopy(model.state_dict())
	progress_bar.close()
	# plot the train and test accuracy with red and blue lines respectively

	plt.plot(range(1, num_epochs + 1), train_accuracies, 'r', label='Train accuracy')
	plt.plot(range(1, num_epochs + 1), test_accuracies, 'b', label='Test accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig(plot_file)
	# clear the plot
	plt.clf()
	return best_weights, best_accuracy


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='MLP classification')
	parser.add_argument('--train_percent', type=float, default=0.3, help='Percentage of data to use for training')
	# batch size
	parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
	# number of epochs
	parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
	# load saved models
	parser.add_argument('--load_saved_models', type=bool, default=False, help='Load saved models')
	# learning rates
	parser.add_argument('--lrs', type=str, default="[1e-6, 1e-5, 1e-4, 1e-3, 1e-2]", help='Learning rates')

	train_percent = parser.parse_args().train_percent
	num_epochs = parser.parse_args().num_epochs
	load_saved_models = parser.parse_args().load_saved_models
	batch_size = parser.parse_args().batch_size
	model_path = f'{prj_path}/models/MLP/classification/best_model_{train_percent}.pt'
	result_path = f'{prj_path}/results/MLP/classification/classification_{train_percent}.csv'
	# create folders if they don't exist
	if not os.path.exists(f'{prj_path}/models/MLP/classification/'):
		os.makedirs(f'{prj_path}/models/MLP/classification/')
	if not os.path.exists(f'{prj_path}/results/MLP/classification/'):
		os.makedirs(f'{prj_path}/results/MLP/classification/')
	if train_percent == -1:
		train_shapes = [
			'euc', 
			'hyperboloid', 
			'poincare', 
			'S2', 
			'S5',
			'sigma.001',
			'sigma.03',
		]
		test_shapes = [
			'S3', 
			'S7',
			'sigma.01',
			'sigma.003',
			'torus'
		]
		print(f'{"-"*20} Using train shapes: {train_shapes} {"-"*20}')
		print(f'{"-"*20} Using test shapes: {test_shapes} {"-"*20}')
		train_dl = prepare_datasets(train_shapes, train_percent, batch_size)
		test_dl = prepare_datasets(test_shapes, train_percent, batch_size)
		# print size of train and test datasets
		print(f'{"-"*20} Train dataset size: {len(train_dl.dataset)} {"-"*20}')
		print(f'{"-"*20} Test dataset size: {len(test_dl.dataset)} {"-"*20}')
	else:
		shapes = [
			'euc', 
			'hyperboloid', 
			'poincare', 
			'S2', 
			'S3', 
			'S5', 
			'S7', 
			'sigma.01', 
			'sigma.001',
			'sigma.03',
			'sigma.003', 
			'torus'
		]
		print(f'{"-"*20} Training using {train_percent*100}% of all data {"-"*20}')
		train_dl, test_dl = prepare_datasets(shapes, train_percent, batch_size)

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	print(f'{"-"*20} Using device: {device} {"-"*20}')

	model = nn.Sequential(
		nn.Linear(2500, 1024),
		nn.ReLU(),
		nn.Linear(1024, 512),
		nn.ReLU(),
		nn.Linear(512, 256),
		nn.ReLU(),
		nn.Linear(256, 3)
	)
	initial_weights = copy.deepcopy(model.state_dict())
	accuracy_fun = nn.CrossEntropyLoss()

	lrs = eval(parser.parse_args().lrs)

	if not load_saved_models:
		best_weights = None
		best_accuracy = -np.inf
		best_lr = 0
		with open(result_path, 'w') as f:
			print(f'lr,accuracy', file=f)
			for idx, lr in enumerate(lrs):
				if device.type == 'cuda':
					torch.cuda.empty_cache()
				model.load_state_dict(initial_weights)
				optimizer = torch.optim.Adam(model.parameters(), lr=lr)
				model = model.to(device)
				plot_file = f'{prj_path}/results/MLP/classification/accuracy_{train_percent}_{lr}.png'
				detailed_result_path = f'{prj_path}/results/MLP/classification/detailed_result_{train_percent}_{lr}.csv'
				weights, accuracy = train_and_test(model, train_dl, test_dl, optimizer, accuracy_fun, device, num_epochs, plot_file, detailed_result_path)
				print(f'Accuracy for lr={lr}: {accuracy:.3f}%')
				print(f'{lr:10.6f},{accuracy:10.3f}%', file=f)
				if idx == 0 or accuracy > best_accuracy:
					best_accuracy = accuracy
					best_weights = weights
					best_lr = lr

		best_model = model
		best_model.load_state_dict(best_weights)
		print(f'Best accuracy for lr={best_lr}: {best_accuracy:.3f}%')
		torch.save(best_weights, model_path)
	else:
		# load the saved model
		model.load_state_dict(torch.load(model_path))
		model = model.to(device)
		test_accuracy = test(model, test_dl, device)
		best_model = model
		print(f'Best model test accuracy: {test_accuracy:.3f}%')
		train_accuracy = test(model, train_dl, device)
		print(f'Best model train accuracy: {train_accuracy:.3f}%')

	# test 20 random samples using the best model and print the results
	random_indices = np.random.choice(len(test_dl), 20)
	for i in random_indices:
		# print in one line
		X, y = test_dl.dataset[i]
		X = X.to(device)
		y = y.to(device)
		outputs = best_model(X)
		#         _, predicted = torch.max(outputs.data, 1)
		# IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
		_, predicted = torch.max(outputs.data, 0)
		# convert the labels to 0->-, 1->0, 2->+
		y = '-' if y == 0 else ('0' if y == 1 else '+')
		predicted = '-' if predicted == 0 else ('0' if predicted == 1 else '+')
		print(f'predicted: {predicted}, actual: {y}')



