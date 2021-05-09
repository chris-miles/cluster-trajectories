# %%
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


def make_zHatTrue(trueLabels,nClusters):
    """
    make_zHatTrue, helper function that returns a *matrix* where each row has a 1 in the true cluster position 

    inputs: trueLabels: Nx1 array of true cluster labels, e.g. [0,1,2,...]
            nClusters: # of clusters

    output: zTrue=NxnCluster matrix
    """ 
    N = len(trueLabels)
    zTrue = np.zeros([N,nClusters])
    for c in range(N):
        zTrue[c, trueLabels[c]] = 1   
    return zTrue
   
def find_best_clustermatch(trueLabels, zHat, muHat, qHat, pHat):
    """
    find_best_clustermatch uses the Hungarian algorithm to find the permutation of zHat that minimizes the error with the true Z

    inputs: trueLabels: N x 1 array of true cluster labels, e.g. [0,1,2,...]
            zHat: N x nCluster matrix of estimated Z probabilities
            muHat, qHat, pHat: all estimates from EM, no effect on algorithm

    outputs: trueLabels = nCluster x 1 array of optimal permutation,
             zHat_c, qHat_c, muHat_c, pHat_c = original versions permuted to optimal 
    """     
    nClusters = np.shape(zHat)[1]
    N = np.shape(zHat)[0]
    zTrue = make_zHatTrue(trueLabels,nClusters)

    cost= np.matmul(np.transpose(zTrue),zHat)
    _, col_ind = linear_sum_assignment(cost,maximize=True)
    best_assignment = col_ind

    zHat_c = np.zeros(np.shape(zHat))
    muHat_c = np.zeros(np.shape(muHat))
    qHat_c = np.zeros(np.shape(qHat))
    pHat_c = np.zeros(np.shape(pHat))
    for c in range(nClusters):
        zHat_c[:,c] = zHat[:,best_assignment[c]]
        qHat_c[c,:] = qHat[best_assignment[c],:]
        muHat_c[c] = muHat[best_assignment[c]]
        pHat_c[c,:,:] = pHat[best_assignment[c],:,:]
    
    return best_assignment, zHat_c, qHat_c, muHat_c, pHat_c


def markov_next_state(transition_matrix, current_state):
    """
    markov_next_state is a helper function to simulate next state of Markov chain

    inputs: transition_matrix: nStates x nStates matrix 
            current_state: # from 0... nStates-1 

    outputs: next_state: # from 0...nStates-1, sampled from transition_matrix
    """      
    num_states = np.shape(transition_matrix)[1]
    states = np.arange(num_states)
    return np.random.choice(
        states, 
        p=transition_matrix[current_state, :]
    )

def markov_generate_states(transition_matrix,current_state, T):
    """
    markov_generate_states is a helper function to simulate a sequence from a Markov chain

    inputs: transition_matrix: nStates x nStates matrix 
            current_state: # from 0... nStates-1, initial state for trajectory
            T: number of samples to generate for this trajectory 

    outputs: future_states, T x 1 array of samples from Markov chain
    """        
    future_states = []
    for i in range(T):
        next_state = markov_next_state(transition_matrix,current_state)
        future_states.append(next_state)
        current_state = next_state
    return future_states

def prob_of_trajectory(transition_matrix, sequence):
    """
    prob_of_trajectory is a helper function to compute probability of a trajectory given a transition matrix

    inputs: transition_matrix =  nStates x nStates matrix 
            sequence =  T  x 1 sequence of states 

    outputs: p = product of P[i,i+1] for this sequence
    """     
    p=1
    for i in range(len(sequence)-1):
        state_i = np.int_(sequence[i])
        state_ii = np.int_(sequence[i+1])
        p_transition=transition_matrix[state_i,state_ii]
        p = p*p_transition
    return p     

def logprob_of_trajectory(transition_matrix, sequence):
    """
    logprob_of_trajectory is a helper function to compute LOG probability of a trajectory given a transition matrix

    inputs: transition_matrix =  nStates x nStates matrix 
            sequence =  T  x 1 sequence of states 

    outputs: logp = sum log(P[i,i+1]) for this sequence
    """          
    logp = 0
    for i in range(len(sequence)-1):
        state_i = np.int_(sequence[i])
        state_ii = np.int_(sequence[i+1])
        logp_transition=np.log(transition_matrix[state_i,state_ii])
        logp = logp+logp_transition
    return logp       

def transition_counts(sequence,nStates):
    """
    transition_counts is a helper function to count the frequency of TRANSITIONS in a sequence from a Markov Chain

    inputs: sequence =  T  x 1 sequence of observations from a Markov chain  
            nStates  =  # assumed number of states in the chain 

    outputs: counts = nStates x nStates, total occurences of each transition. ijth element is count i->j
    """     
    counts = np.zeros([nStates,nStates])
    for i in range(len(sequence)-1):
        state_i = np.int_(sequence[i])
        state_ii = np.int_(sequence[i+1])
        counts[state_i, state_ii] = counts[state_i, state_ii]+1 
    return counts

def trajectory_counts(sequence,nStates):
    """
    trajectory_counts is a helper function to count the frequency of observation of each state in a sequence

    inputs: sequence =  T  x 1 sequence of observations from a Markov chain  
            nStates  =  # assumed number of states in the chain 

    outputs: counts = total occurences of each state in the sequence
    """     
    counts = np.zeros(nStates,)
    for i in range(len(sequence)):
        state_i = np.int_(sequence[i])
        counts[state_i] = counts[state_i]+1 
    return counts

def eStep(X, muHat, qHat, pHat, M, nStates):
    """
    eStep does the E step in our EM algorithm. If trajectories are very long P[trajectory] is VERY TINY, so we need to do the logsum trick, see here 
    https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/ 

    inputs: X = N x T matrix, N trajectories of length T, 
            muHat = nClusters x 1, estimated mu, mixture probs 
            qHat = nClusters x nStates, estimated initial probs
            pHat = nClusters x nStates x nStates, estimated transition matrices
            nStates  =  # assumed number of states in the chain, alphabet size

    outputs: zHat = updated estimates of probability of each cluster
    """     
    np.seterr(divide = 'ignore')
    N = np.shape(X)[0] 
    T = np.shape(X)[1]
    zHat = np.zeros([N,M])


    for n in range(N):
        Xn = X[n,:]
        lognumerators = np.zeros(M,)
        for i in range(M):
            mu_i = muHat[i]
            X0 = np.int_(Xn[0])
            qiX0 = qHat[i,X0]
            logprodpi = logprob_of_trajectory(pHat[i,:,:],Xn)
            lognumerators[i] = np.log(mu_i)+np.log(qiX0)+logprodpi
            zz = np.exp(lognumerators-logsumexp(lognumerators)) # logsum trick here

        zHat[n,:] = zz
    return zHat


def mStep(X, zHat, M, nStates):
    """
    mStep  does the M step in our EM algorithm. Probably a better way to do this computationally.. Have *NOT* implemented logsum trick here. 

    inputs: X = N x T matrix, N trajectories of length T, 
            zHat = updated estimates of probability of each cluster
            M = #, number of clusters
            nStates  =  # assumed number of states in the chain, alphabet size

    outputs: muHat = nClusters x 1, estimated mu, mixture probs 
            qHat = nClusters x nStates, estimated initial probs
            pHat = nClusters x nStates x nStates, estimated transition matrices
    """     
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

    denom = zHat.sum(axis=0)[:,None]
    qHat = np.divide(numerator,denom,out=np.zeros_like(numerator), where=numerator!=0)

    # I don't think looping over clusters first here is ideal because you have to recompute for every trajectory but this is easiest.. if too slow can fix later  
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
        denominator = denom[:,None]
        pimat = np.divide(numerator,denominator,out=np.zeros_like(numerator), where=numerator!=0)
        pHat[i,:,:] = pimat



    return muHat, qHat, pHat
    

def doEM(X, M, nStates, tol):
    """
    doEM performs our EM algorithm for a trajectory, stopping when the next step hits a level of tolerance.

    Initial condition is for each trajectory: pick k~unif(nClusters)
    and assign P[Z=k] = 1. Could try others, unsure if this is ideal

    inputs: X = N x T matrix, N trajectories of length T, 
            M = #, number of clusters
            tol  =  absolute err level (in 2-norm), of when Z, mu, q stop changing by this amount, halt the algorithm


    outputs: zHat = updated estimates of probability of each cluster
            muHat = nClusters x 1, estimated mu, mixture probs 
            qHat = nClusters x nStates, estimated initial probs
            pHat = nClusters x nStates x nStates, estimated transition matrices
            steps = # of steps EM alg took
    """     
    # assign initial Z
    N = np.shape(X)[0] 
    zHat = np.zeros([N,M])
    for n in range (N):
        random_initial_assign = np.random.choice(M)
        zHat[n,random_initial_assign]=1

    # initialize tolerances
    steps = 0
    zHatdiff = 1.0
    qHatdiff = 1.0
    muHatdiff = 1.0

    
    while zHatdiff>tol and muHatdiff>tol and qHatdiff >tol:
        steps = steps+1
        muHatnew, qHatnew, pHat = mStep(X,zHat,M, nStates)

        if steps > 1: # no values of mHat, qHat to reference first update
            muHatdiff = np.linalg.norm(muHatnew-muHat,ord=2)
            qHatdiff = np.linalg.norm(qHatnew-qHat,ord=2)

        muHat = muHatnew
        qHat = qHatnew 

        zHatnew = eStep(X,muHat, qHat, pHat, M, nStates)
        zHatdiff = np.linalg.norm(zHatnew-zHat,ord=2)
        zHat = zHatnew
    return zHat, muHat, qHat, pHat, steps
    
def generateChains(nStates, nClusters):
    """
    generateChains randomly generates Markov chains for testing, also generates initial probability distributions, q

    Randomness is nothing sophisticated, generate each row from unif([0,1])
    and normalize so it is a transition matrix.

    inputs: nStates = #, alphabet size, labeled 0 thru nStates-1
            nClusters = #, number of distinct chains to generate


    outputs: transition_matrices = nClusters x nStates x nStates, randomly generate transition matrices
    initDists = nClusters x nStates = q, initial probability densities
, initDists
    """    
    transition_matrices = np.zeros([nClusters, nStates,nStates])
    initDists = np.zeros([nClusters, nStates]) 

    for i in range(nClusters):
        matrix = np.random.rand(nStates,nStates)
        stoch_matrix = matrix/matrix.sum(axis=1)[:,None]
        transition_matrices[i,:,:]  = stoch_matrix
        initDist = np.random.rand(nStates,)
        initDist_normal = initDist/initDist.sum()
        initDists[i,:] = initDist_normal

    return transition_matrices, initDists


def generateTrajectories(N, T, mixtureProbs, initDists, transition_matrices,nClusters):
    """
    generateTrajectories is for testing, samples a given mixture of Markov chains

    inputs: N = # of trajectories
            T = # of time points on each trajectory
            mixtureProbs = N x 1,  probabilities of each trajectory falling into a cluster
            initDists = nClusters x nStates, q, initial probability densities
            transition_matrices = nClusters x nStates x xNstates 


    outputs: X = N x T, trajectories from alphabet nStates 
             trueLabels = N x 1,  true cluster labels for each trajectory
    """    
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


