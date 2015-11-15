from kmeans import *
import sys
import matplotlib.pyplot as plt
plt.ion()

def mogEM(x, K, iters, minVary=0, use_kmeans=False):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape

  # Initialize the parameters
  randConst = 10
  p = randConst + np.random.rand(K, 1)
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
 
  # Change the initializaiton with Kmeans here
  #--------------------  Add your code here --------------------  
  mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)

  if use_kmeans:
      # initialize with kmeans with 5 iterations
      mu = KMeans(x, K, 5)
  
  #------------------------------------------------------------  
  vary = vr * np.ones((1, K)) * 2
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary

  logProbX = np.zeros((iters, 1))

  # Do iters iterations of EM
  for i in xrange(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    respX = np.zeros((N, K))
    respDist = np.zeros((N, K))
    logProb = np.zeros((1, T))
    ivary = 1 / vary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
    logPcAndx = np.zeros((K, T))
    for k in xrange(K):
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px
    logProb = np.log(Px) + mx
    logProbX[i] = np.sum(logProb)

    print 'Iter %d logProb %.5f' % (i, logProbX[i])

    # Plot log prob of data
    plt.figure(1);
    plt.clf()
    plt.plot(np.arange(i), logProbX[:i], 'r-')
    plt.title('Log-probability of data versus # iterations of EM')
    plt.xlabel('Iterations of EM')
    plt.ylabel('log P(D)');
    plt.draw()

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in xrange(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

    # Do the M step
    p = respTot
    mu = respX / respTot.T
    vary = respDist / respTot.T
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary
  
  return p, mu, vary, logProbX

def mogLogProb(p, mu, vary, x):
  """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
  K = p.shape[0]
  N, T = x.shape
  ivary = 1 / vary
  logProb = np.zeros(T)
  for t in xrange(T):
    # Compute log P(c)p(x|c) and then log p(x)
    logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
        - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
        - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)**2, axis=0).reshape(-1, 1)

    mx = np.max(logPcAndx, axis=0)
    logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
  return logProb

def q2():
  iters = 10
  minVary = 0.01
  nCluster = 2

  inputs_train2, _, _, _, _, _ =  LoadData('digits.npz', True, False)
  inputs_train3, _, _, _, _, _ = LoadData('digits.npz', False, True)
  p2, mu2, var2, logProbX2 = mogEM(inputs_train2, nCluster, iters, minVary)
  p3, mu3, var3, logProbX3 = mogEM(inputs_train3, nCluster, iters, minVary)
  
  ShowMeans(mu2, title='MoG_clustering_result_for_2_mean')
  ShowMeans(mu3, title='MoG_clustering_result_for_3_mean')

  ShowMeans(var2, title='MoG_clustering_result_for_2_var')
  ShowMeans(var3, title='MoG_clustering_result_for_3_var')

  print "LogProbX for 2 is: ", logProbX2[-1]
  print "Mixing for 2 is: ", p2
  print "************************************"
  print "LogProbX for 3 is: ", logProbX3[-1]
  print "Mixing for 3 is: ", p3

def q3():
  iters = 10
  minVary = 0.01
  nCluster = 20
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  p, mu, var, logProbX = mogEM(inputs_train, nCluster, iters, minVary, use_kmeans=False)
  print "LogProbX without kmeans is: ", logProbX
 
  p, mu, var, logProbX = mogEM(inputs_train, nCluster, iters, minVary, use_kmeans=True)
  print "LogProbX with kmeans is: ", logProbX
  #------------------- Add your code here ---------------------

  raw_input('Press Enter to continue.')

def q4():
  iters = 10
  minVary = 0.01
  numComponents = np.array([2, 5, 15, 25])
  T = numComponents.shape[0]  

  errorTrain = np.zeros(T)
  errorTest = np.zeros(T)
  errorValidation = np.zeros(T)
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  
  for t in xrange(T): 
    K = numComponents[t]
    # Train a MoG model with K components for digit 2
    #-------------------- Add your code here --------------------------------
    p2, mu2, var2, logProbX2 = mogEM(train2, K, iters, minVary, use_kmeans=True)

    
    # Train a MoG model with K components for digit 3
    #-------------------- Add your code here --------------------------------
    p3, mu3, var3, logProbX3 = mogEM(train3, K, iters, minVary, use_kmeans=True)

    
    # Caculate the probability P(d=1|x) and P(d=2|x),
    # classify validation set, and compute error rate

    # comptue P(d=1|x), a vector of lenght k representing the probability of 
    # P(d=1_i|x) for i in [1,...,K]
    p_d1_valid = mogLogProb(p2, mu2, var2, inputs_valid)
    p_d2_valid = mogLogProb(p3, mu3, var3, inputs_valid)

    p_d1_train = mogLogProb(p2, mu2, var2, inputs_train)
    p_d2_train = mogLogProb(p3, mu3, var3, inputs_train)

    p_d1_test = mogLogProb(p2, mu2, var2, inputs_test)
    p_d2_test = mogLogProb(p3, mu3, var3, inputs_test)

    # classified as '3' iff p_d2 >= p_d1
    decision_train = p_d2_train >= p_d1_train
    decision_valid = p_d2_valid >= p_d1_valid
    decision_test = p_d2_test >= p_d1_test
    # correct_valid[i] == 1 iff decision_valid[i] == target_valid[i], 0 otherwise
    correct_train = decision_train == target_train
    correct_valid = decision_valid == target_valid
    correct_test = decision_test == target_test
    # perc_error = 1 - perc_correct
    errorValidation[t] = 1.0 - correct_valid.mean()
    errorTrain[t] = 1.0 - correct_train.mean()
    errorTest[t] = 1.0 - correct_test.mean()
    
  # Plot the error rate
  print "Train: ", errorTrain
  print "Validation: ", errorValidation
  print "Test: ", errorTest
  plt.clf()
  #-------------------- Add your code here --------------------------------
  plt.title("MoG_number_of_components")
  plt.plot(numComponents, errorTrain, 'r', label='Train_perc_error')
  plt.plot(numComponents, errorValidation, 'g', label='Validation_perc_error')
  plt.plot(numComponents, errorTest, 'b', label='Test_perc_error')
  plt.legend()
  plt.xlabel('Number of components in MoG')
  plt.ylabel('Classification error');
  plt.draw()
  plt.savefig("MoG_num_component_vs_error")
  raw_input('Press Enter to continue.')

def q5():
  # Choose the best mixture of Gaussian classifier you have, compare this
  # mixture of Gaussian classifier with the neural network you implemented in
  # the last assignment.
  iters = 10
  minVary = 0.01
  errorTrain = np.zeros(4)
  errorTest = np.zeros(4)
  errorValidation = np.zeros(4)
  K = 25
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)

  # Train model with K components
  p2, mu2, var2, logProbX2 = mogEM(train2, K, iters, minVary, use_kmeans=True)
  p3, mu3, var3, logProbX3 = mogEM(train3, K, iters, minVary, use_kmeans=True)

  p_d1_valid = mogLogProb(p2, mu2, var2, inputs_valid)
  p_d2_valid = mogLogProb(p3, mu3, var3, inputs_valid)
  
  p_d1_train = mogLogProb(p2, mu2, var2, inputs_train)
  p_d2_train = mogLogProb(p3, mu3, var3, inputs_train)
  
  p_d1_test = mogLogProb(p2, mu2, var2, inputs_test)
  p_d2_test = mogLogProb(p3, mu3, var3, inputs_test)
  
  # classified as '3' iff p_d2 >= p_d1
  decision_train = p_d2_train >= p_d1_train
  decision_valid = p_d2_valid >= p_d1_valid
  decision_test = p_d2_test >= p_d1_test
  # correct_valid[i] == 1 iff decision_valid[i] == target_valid[i], 0 otherwise
  correct_train = decision_train == target_train
  correct_valid = decision_valid == target_valid
  correct_test = decision_test == target_test
  # perc_error = 1 - perc_correct
  print "Validation error: ", 1.0 - correct_valid.mean()
  print "Train error: ", 1.0 - correct_train.mean()
  print "Test error: ", 1.0 - correct_test.mean()

  # Train neural network classifier. The number of hidden units should be
  # equal to the number of mixture components.
  import nn
  eps = 0.2
  momentum = 0.5
  num_epochs = 1000
  (
      W1, W2, b1, b2,
      train_error, valid_error, test_error,
      train_class_error, valid_class_error, test_class_error,
  ) = nn.TrainNN(K, eps, momentum, num_epochs, run_test=True)

  print "NN Train error: ", train_class_error[-1]
  print "NN Validation error: ", valid_class_error[-1]
  print "NN Test error: ", test_class_error[-1]

  # Show the error rate comparison.
  #-------------------- Add your code here --------------------------------

  raw_input('Press Enter to continue.')

if __name__ == '__main__':
  q2() 
  q3()
  #q4()
  #q5()
