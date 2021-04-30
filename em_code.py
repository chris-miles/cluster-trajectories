import numpy as np
import random as rm

def markov_next_state(transition_matrix, current_state):
    num_states = np.shape(transition_matrix)[1]
    states = np.arange(num_states)
    return np.random.choice(
        states, 
        p=transition_matrix[current_state, :]
    )

def markov_generate_states(transition_matrix,current_state, T):
    future_states = []
    for i in range(T):
        next_state = markov_next_state(transition_matrix,current_state)
        future_states.append(next_state)
        current_state = next_state
    return future_states

def prob_of_trajectory(transition_matrix, sequence):
    p = 1
    for i in range(len(sequence)-1):
        state_i = np.int_(sequence[i])
        state_ii = np.int_(sequence[i+1])
        p_transition=transition_matrix[state_i,state_ii]
        p = p*p_transition
    return p     

def transition_counts(sequence,nStates):
    counts = np.zeros([nStates,nStates])
    for i in range(len(sequence)-1):
        state_i = np.int_(sequence[i])
        state_ii = np.int_(sequence[i+1])
        counts[state_i, state_ii] = counts[state_i, state_ii]+1 
    return counts

def trajectory_counts(sequence,nStates):
    counts = np.zeros(nStates,)
    for i in range(len(sequence)):
        state_i = np.int_(sequence[i])
        counts[state_i] = counts[state_i]+1 
    return counts


def eStep(X, muHat, qHat, pHat, M, nStates):
    # assumes X is an n x T matrix, so all trajectories have same length 
    # m is the number of clusters 
    N = np.shape(X)[0] 
    T = np.shape(X)[1]
    zHat = np.zeros([N,M])

    # assuming muHat: M x 1 
    #              qhat:  M x nStates
    #              phat = M x nStates x nStates

    for n in range(N):
        Xn = X[n,:]
        # smartest way to do this is compute all of the numerators, and then divide by their sum
        numerators = np.zeros(M,)
        for i in range(M):
            mu_i = muHat[i]
            X0 = np.int_(Xn[0])
            qiX0 = qHat[i,X0]
            prodpi = prob_of_trajectory(pHat[i,:,:],Xn)
            numerators[i] = mu_i*qiX0*prodpi
        zz = numerators/numerators.sum()
        zHat[n,:] = zz
    return zHat

def mStep(X, zHat, M, nStates):
    # assumes X is an n x T matrix, so all trajectories have same length 
    # m is the number of clusters 
    # zHat is N x M
    N = np.shape(X)[0] 
    T = np.shape(X)[1]

    muHat = np.sum(zHat,axis=0)/N

    qHat = np.zeros([M, nStates])
    pHat = np.zeros([M, nStates, nStates])
    
    # qHat computation, maybe not the best way but optimize later
    numerator = np.zeros([M,nStates])
    for n in range(N):
        Zn = zHat[n,:]
        X0 = np.int_(X[n,0])
        indicator = np.zeros(nStates,)
        indicator[X0] = 1
        numerator = numerator + np.outer(Zn,indicator)

    qHat = numerator/(zHat.sum(axis=0))[:,None]

    # do something like this for phat too but its harder...
    # I don't think looping over i first makes sense because you have to recompute for every trajectory but this is easiest.. if too slow can fix later  
    for i in range(M):
        numerator = np.zeros([nStates,nStates])
        denom = np.zeros(nStates,)
        for n in range(N):
            Xn = X[n,:]
            PZni = zHat[n,i]
            trans_counts = transition_counts(Xn,nStates)
            numerator = numerator+PZni*trans_counts 
            traj_counts = trajectory_counts(Xn[:-1],nStates)
            denom = denom+PZni*traj_counts
        pimat = np.divide(numerator, denom[:,None])
        pHat[i,:,:] = pimat



    return muHat, qHat, pHat
    

def doEM(X, M, nStates, S):
    # M = nClusters = number of clusters
    # S = number of EM steps
    # X is data matrix
    # easiest is either to assign Zhat^n = uniform, 
    # or I_{k}, k chosen randomly. doing this latter
    N = np.shape(X)[0] 
    zHat = np.zeros([N,M])
    for n in range (N):
        random_initial_assign = np.random.choice(M)
        zHat[n,random_initial_assign]=1

     # then, do M, then E, repeat for S steps, S can be chosen better
    for s in range(S):
        muHat, qHat, pHat = mStep(X,zHat,M, nStates)
        zHat = eStep(X,muHat, qHat, pHat, M, nStates)
    return zHat, muHat, qHat, pHat
    
def generateChains(nStates, nClusters):
    # transition_matrices is nClusters x nStates x nStates 
    # generating singly stochastic matrices is easy -- just generate rows and make sure they sum to 1 
    transition_matrices = np.zeros([nClusters, nStates,nStates])
    initDists = np.zeros([nClusters, nStates])  # randomly choose these too I guess? 

    for i in range(nClusters):
        matrix = np.random.rand(nStates,nStates)
        stoch_matrix = matrix/matrix.sum(axis=1)[:,None]
        transition_matrices[i,:,:]  = stoch_matrix
        initDist = np.random.rand(nStates,)
        initDist_normal = initDist/initDist.sum()
        initDists[i,:] = initDist_normal

    return transition_matrices, initDists


def generateTrajectories(N, T, mixtureProbs, initDists, transition_matrices,nClusters):
    X = np.zeros([N,T])
    nStates = np.shape(transition_matrices)[2]
    trueLabels = np.random.choice(nClusters, N, p = mixtureProbs)
    for n in range(N):
        trueLabeln = trueLabels[n]
        X0 = np.random.choice(nStates, p=initDists[trueLabeln,:])
        Xn = markov_generate_states(transition_matrices[trueLabeln,:,:],X0,T)
        X[n,:] = Xn
    # transition_matrices is nClusters x nStates x nStates 
    return X,  trueLabels






# tt,qq=generateChains(3,2)
# X,truelabels = generateTrajectories(10, 100,[0.5,0.5], qq, tt, 2)
#zHat, muHat, qHat, pHat = doEM(X, nClusters, nStates, 100)
