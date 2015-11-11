from util import *
import sys
import matplotlib.pyplot as plt
plt.ion()

class DimensionMismatchError(Exception):
    pass

def InitNN(num_inputs, num_hiddens, num_outputs):
  """Initializes NN parameters."""
  W1 = 0.01 * np.random.randn(num_inputs, num_hiddens)
  W2 = 0.01 * np.random.randn(num_hiddens, num_outputs)
  b1 = np.zeros((num_hiddens, 1))
  b2 = np.zeros((num_outputs, 1))
  return W1, W2, b1, b2

def TrainNN(num_hiddens, eps, momentum, num_epochs, run_test=False):
  """Trains a single hidden layer NN.

  Inputs:
    num_hiddens: NUmber of hidden units.
    eps: Learning rate.
    momentum: Momentum.
    num_epochs: Number of epochs to run training for.

  Returns:
    W1: First layer weights.
    W2: Second layer weights.
    b1: Hidden layer bias.
    b2: Output layer bias.
    train_error: Training error at at epoch.
    valid_error: Validation error at at epoch.
  """

  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  W1, W2, b1, b2 = InitNN(inputs_train.shape[0], num_hiddens, target_train.shape[0])
  dW1 = np.zeros(W1.shape)
  dW2 = np.zeros(W2.shape)
  db1 = np.zeros(b1.shape)
  db2 = np.zeros(b2.shape)
  train_error = []
  valid_error = []
  test_error = []
  train_class_error = []
  valid_class_error = []
  test_class_error = []
  num_train_cases = inputs_train.shape[1]
  for epoch in xrange(num_epochs):
    # Forward prop
    h_input = np.dot(W1.T, inputs_train) + b1  # Input to hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction.

    # Compute cross entropy
    train_CE = -np.mean(target_train * np.log(prediction) + (1 - target_train) * np.log(1 - prediction))

    # Calculate classification error
    train_Err = get_error_perc(target_train, prediction) 
    train_class_error.append(train_Err)
    
    # Compute deriv
    dEbydlogit = prediction - target_train

    # Backprop
    dEbydh_output = np.dot(W2, dEbydlogit)
    dEbydh_input = dEbydh_output * h_output * (1 - h_output)

    # Gradients for weights and biases.
    dEbydW2 = np.dot(h_output, dEbydlogit.T)
    dEbydb2 = np.sum(dEbydlogit, axis=1).reshape(-1, 1)
    dEbydW1 = np.dot(inputs_train, dEbydh_input.T)
    dEbydb1 = np.sum(dEbydh_input, axis=1).reshape(-1, 1)

    #%%%% Update the weights at the end of the epoch %%%%%%
    dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1
    dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2
    db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1
    db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2

    W1 = W1 + dW1
    W2 = W2 + dW2
    b1 = b1 + db1
    b2 = b2 + db2

    valid_CE = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)

    # get classification error
    valid_Err = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2, output_Err=True)
    valid_class_error.append(valid_Err)

    # generalize -- run test set as well
    if run_test:
        test_CE = Evaluate(inputs_test, target_test, W1, W2, b1, b2)
        test_Err = Evaluate(inputs_test, target_test, W1, W2, b1, b2, output_Err=True)
        test_error.append(test_CE)
        test_class_error.append(test_Err)


    train_error.append(train_CE)
    valid_error.append(valid_CE)
    sys.stdout.write('\rStep %d Train CE %.5f Validation CE %.5f' % (epoch, train_CE, valid_CE))
    sys.stdout.flush()
    if (epoch % 100 == 0):
      sys.stdout.write('\n')

  sys.stdout.write('\n')
  final_train_error = Evaluate(inputs_train, target_train, W1, W2, b1, b2)
  final_valid_error = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)
  final_test_error = Evaluate(inputs_test, target_test, W1, W2, b1, b2)
  print 'Error: Train %.5f Validation %.5f Test %.5f' % (final_train_error, final_valid_error, final_test_error)
  if run_test:
    return W1, W2, b1, b2, train_error, valid_error, test_error, train_class_error, valid_class_error, test_class_error
  return W1, W2, b1, b2, train_error, valid_error, train_class_error, valid_class_error

def Evaluate(inputs, target, W1, W2, b1, b2, output_Err=False):
  """Evaluates the model on inputs and target."""
  h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
  h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
  logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
  prediction = 1 / (1 + np.exp(-logit))  # Output prediction.
  CE = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
  if output_Err:
      return get_error_perc(target, prediction)
  return CE

def DisplayErrorPlot(train_error, valid_error, mode="Cross Entropy", test=None):
  plt.figure(1)
  plt.clf()
  plt.plot(range(len(train_error)), train_error, 'b', label='Train')
  plt.plot(range(len(valid_error)), valid_error, 'g', label='Validation')
  if test is not None:
      plt.plot(range(len(test)), test, 'r', label='Test')
  plt.xlabel('Epochs')
  plt.ylabel(mode)
  plt.legend()
  #plt.show()
  plt.savefig("nn_10_layer_" + mode)
  #raw_input('Press Enter to exit.')

def SaveModel(modelfile, W1, W2, b1, b2, train_error, valid_error):
  """Saves the model to a numpy file."""
  model = {'W1': W1, 'W2' : W2, 'b1' : b1, 'b2' : b2,
           'train_error' : train_error, 'valid_error' : valid_error}
  print 'Writing model to %s' % modelfile
  np.savez(modelfile, **model)

def LoadModel(modelfile):
  """Loads model from numpy file."""
  model = np.load(modelfile)
  return model['W1'], model['W2'], model['b1'], model['b2'], model['train_error'], model['valid_error']

def main():
  num_hiddens = 10
  eps = 0.1
  momentum = 0.0
  num_epochs = 1000

  current_problem = [2.4]
  print "Running problems: ", current_problem

  # 2.1 and 2.2
  if 2.2 in current_problem and 2.1 in current_problem:
      W1, W2, b1, b2, train_error, valid_error, train_class_error, valid_class_error = TrainNN(num_hiddens, eps, momentum, num_epochs)
      DisplayErrorPlot(train_error, valid_error, mode='cross_entropy')
      DisplayErrorPlot(train_class_error, valid_class_error, mode='classification_error') 

  # 2.3
  if 2.3 in current_problem:
      for eps in [0.5, 0.2, 0.1]:
          W1, W2, b1, b2, train_error, valid_error, train_class_error, valid_class_error = TrainNN(num_hiddens, eps, momentum, num_epochs)
          # iterate through different eps
          suffix = '_at_eps_' + str(eps)
          suffix = suffix.replace('.', ',')
          DisplayErrorPlot(train_error, valid_error, mode='cross_entropy' + suffix)
          DisplayErrorPlot(train_class_error, valid_class_error, mode='classification_error' + suffix) 

      for momentum in [0.9, 0.5, 0.0]:
          W1, W2, b1, b2, train_error, valid_error, train_class_error, valid_class_error = TrainNN(num_hiddens, eps, momentum, num_epochs)
          # iterate through different eps
          suffix = '_at_momentum_' + str(momentum)
          suffix = suffix.replace('.', ',')
          DisplayErrorPlot(train_error, valid_error, mode='cross_entropy' + suffix)
          DisplayErrorPlot(train_class_error, valid_class_error, mode='classification_error' + suffix) 

  # 2.4
  if 2.4 in current_problem:
      eps = 0.2
      momentum = 0.5
      for num_hiddens in [2, 5, 10, 30, 100]:
          (
              W1, W2, b1, b2,
              train_error, valid_error, test_error,
              train_class_error, valid_class_error, test_class_error,
          ) = TrainNN(num_hiddens, eps, momentum, num_epochs, run_test=True)

          # iterate through different eps
          suffix = '_at_hidden_unit_' + str(num_hiddens)
          DisplayErrorPlot(train_error, valid_error, mode='cross_entropy' + suffix, test=test_error)
          DisplayErrorPlot(train_class_error, valid_class_error, mode='classification_error' + suffix, test=test_class_error) 

  # If you wish to save the model for future use :
  # outputfile = 'model.npz'
  # SaveModel(outputfile, W1, W2, b1, b2, train_error, valid_error)

def get_error_perc(target, prediction):
    if len(target[0]) != len(prediction[0]):
        raise DimensionMismatchError(
                "Dimension of target does not match that of prediction! {0} : {1}"
                .format(len(target), len(prediction))
        )
    n_sample = len(target[0])
    n_correct = 0
    for (t, p) in zip(target[0], prediction[0]):
        if (t == 1 and p >= 0.5) or (t == 0 and p < 0.5):
            n_correct += 1
    return 1.0 - 1.0 * n_correct / n_sample

if __name__ == '__main__':
  main()
