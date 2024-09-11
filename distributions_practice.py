import numpy as np
import matplotlib.pyplot as plt

## Uniform Distribution over k = 5
k = 5
_ = plt.bar(np.arange(1, k + 1), np.ones(k) / k)
plt.show()

## Bernoulli Distribution 
p1 = 0.7
plt.bar([0, 1], [1 - p1, p1])
_ = plt.xticks([0, 1], ['False', 'True'])
plt.show()

## Catergorical Distribution
# sample space: we use a list (rather than a set) because we want it sorted
sample_space = ['blue', 'red', 'white']
# the values our random variable takes on 
rv_values = [1, 2, 3]
# This is the mapping from sample space (colours) to numbers (assignments of X)
rv_mapping = dict(zip(sample_space, rv_values))
# And these are the probabilities associated with each colour in the example
probabilities = [0.3, 0.2, 0.5] 

print('Sample space:', sample_space)
print('Mapping function:', rv_mapping)
for omega in sample_space:
    x = rv_mapping[omega]
    theta_x = probabilities[x - 1]
    print(" outcome %r mapped to %d has probability %.2f" % (omega, x, theta_x))    
    
plt.bar(rv_values, probabilities)
_ = plt.xticks(rv_values, sample_space)
plt.show()

## Representation of a brenoulli variable by storing the bernoulli parameter
class Bernoulli:
    
    def __init__(self, p):
        """
            X ~ Bernoulli(p)
            where p is the probability of the positive class
        """
        if not (0 <= p <= 1):
            raise ValueError('The Bernoulli parameter must be a probability')
        self.p = p
        
    def sample(self):
        """Returns either 1 or 0 depending on a random simulation"""
        u = np.random.uniform()
        # The idea is to now transform this uniform simulation is such a way that with probability p
        #  we get 1 and with probability (1-p) we get a 0
        # For example, imagine p = 0.7 then we want to return 1 on average 70% of the times this 
        #  code is executed. 
        # Since `u` was uniformly generated, it did not bias any particular segment of the interval 
        #  between 0 and 1. Therefore if we take the first 70% of that interval to return 1, we 
        #  simulate the variable correctly. 70% of the interval between 0 and 1 can for example be represented
        #  by the interval between 0 and 0.7 (the p parameter) 
        return 1 if u < self.p else 0
    
    def sample_n(self, N):
        """Returns an array of N samples"""
        return np.array([self.sample() for _ in range(N)])
    
    def pmf(self, value):
        """
        Probability mass function
        value: an assignment of X
        returns: probability value for X=value
        """
        if value == 1:
            return self.p
        elif value == 0:
            return 1. - self.p
        else:
            raise ValueError('Bernoully variables can only take on 1 or 0')
            
    def cdf(self, a):
        """
        Cumulative probability function
            F_X(a) = \sum_{x <= a} P(X=x)
        a: an assignment of X
        returns: total probability over the interval [-inf, a]
        """
        if a == 0:
            return 1. - self.p
        elif a == 1:
            return 1.
        else:
            raise ValueError('Bernoulli variables can only take on 0 or 1')
            
    def __repr__(self):
        return 'Bernoulli(%s)' % self.p

#Now we can declare a bernoulli random variable and use it to sample an assignment of X
X = Bernoulli(0.7)
print('X ~', X)
for x in range(2):
    print(' P(X=%d) = %.2f F_X(%d) = %.2f' % (x, X.pmf(x), x, X.cdf(x)))

for i in range(10):
    x = X.sample()
    print('assignment {} with probability {:.2f}'.format(x, X.pmf(x)) )
    
## the maximum likelihood estimate of the bernoulli parameter is the ratio at whcih we observe X=1
def bernoulli_mle(observations):
    # this is the number of observations we have
    N = len(observations)
    count1 = observations.sum()
    return float(count1) / N

# Check to show that after taking 1000 samples the estimate is indeed close to 0.7 which we used to define X
N = 1000 #estimation only comes close to 0.7 with enough observations
bernoulli_observations = X.sample_n(N)
print(bernoulli_mle(bernoulli_observations))

## Comparing maximum likelihood estimation using N = 1, 10, 100 and 1000
samples_1 = []
samples_10 = []
samples_100 = []
samples_1000 = []

for i in range(100):
    samples_1.append(bernoulli_mle(X.sample_n(1)))
    samples_10.append(bernoulli_mle(X.sample_n(10)))
    samples_100.append(bernoulli_mle(X.sample_n(100)))
    samples_1000.append(bernoulli_mle(X.sample_n(1000)))

data = [samples_1, samples_10, samples_100, samples_1000]
fig1, ax1 = plt.subplots()
ax1.boxplot(data, labels=[1,10,100,1000])
ax1.set_xlabel('Sample size')
ax1.set_ylabel('MLE')
plt.show()

## Extend the same principles to Categorical random variables
class Categorical:
    
    def __init__(self, theta):
        """
            X ~ Categorical(theta_1, ..., theta_K)
            where theta is the parameter vector and theta_k is the probability of the kth class
            recall that theta_k is bound between 0 and 1 and the sum should be exactly 1
        """
        if sum(theta) != 1:
            raise ValueError('Input must be list of probabilities adding up to 1')
        self.theta = theta

        
    def sample(self):
        """Returns a number from 1 to K representing the sampled value"""
        u = np.random.uniform()
        cumulative_theta = 0
        for i in range(len(self.theta)):
            cumulative_theta += self.theta[i]
            if u < cumulative_theta:
                return (i + 1)
        
    
    def sample_n(self, N):
        """Returns a numpy array containing N simulated values"""
        return np.array([self.sample() for _ in range(N)])
    
    def pmf(self, value):
        """
        Probability mass function evaluated at a certain value
        value: an assignment of X
        """
        if type(value)==int and value>0 and value<=len(self.theta)+1:
            return self.theta[value-1]
        else:
            raise ValueError('Categorical variables can not take float numbers, numbers lower than 1 and higher than k')
    
    def cdf(self, a):
        """
        Cumulative probability function
            F_X(a) = \sum_{x <= a} P(X=x)
        a: an assignment of X
        returns: total probability over the interval [-inf, a]
        """
        cumul = 0
        if type(a)==int and a>0 and a<=len(self.theta):
            for i in range(0, a):
                cumul += self.theta[i]
            return cumul
        else:
            raise ValueError('Categorical variables can only take on indexes between 1 and K')
    
    def __repr__(self):
        return 'Categorical(%s)' % self.theta


parameters = [0.1,0.2,0.7]
X = Categorical(parameters)
## TEST OF REPRESENTATION, printing pmf and cdf for all values the random variable can take
print('X ~', X)

for i in range(len(parameters)):
    print(' P(X=%d) = %.2f' % (i+1, X.pmf(i+1)))
    
# The Maximum Likelihood Estimation algorithm
def categorical_mle(observations):
    # Count the nr of observations and sort them from low to high
    N = len(observations)
    observations.sort()
    
    # Put them in a dictionary with K as the key and its number of occurences as the value
    count = {i:observations.count(i) for i in observations}
    
    # Divide the number of occurences of each key by the total number of observations if in dictionary, 
    # if its not in the dict it had no appearances and the mle is 0.
    k = list(range(1, len(parameters)+1))
    mle = []
    for i in range(0, len(k)):
        if i+1 in count:
            divide = count[i+1]/N
            mle.append(divide)
        else:
            mle.append(0)
    return mle

# initializing the categorical observations for different sample sizes
categorical_observations_1 = X.sample_n(1)
categorical_observations_10 = X.sample_n(10)
categorical_observations_100 = X.sample_n(100)
categorical_observations_1000 = X.sample_n(1000)

## Here the MLE's for N=1 get calculated and a list containing all the observations for N = 1
all_observations_N1 = []

## N=1
for i in range(len(parameters)):
    k1 = []
    
    # k1 will hold a list of the length of the number of k's and this list will have as elements lists with the MLE's
    # assigned to k=1 at position 0, k=2 at position 1, k3 at 2 and so on. These lists will be used to to calculate the mean and
    # standard error. Calculates the mle 1000 times and later takes the mean.
    for j in range(1000):
        #for n=1 we calculate the mle's and put them in a list for each K
        categorical_observations_1 = X.sample_n(1)
        kk1 = categorical_mle(list(categorical_observations_1))
        k1.append(kk1[i])
    all_observations_N1.append(k1)

#calculate the means and standard errors for N1
mean1 = []
stder1 = []
for i in range(len(parameters)):
    mean = np.mean(all_observations_N1[i])
    stder = np.std(all_observations_N1[i])
    mean1.append(mean)
    stder1.append(stder)

## N = 10
all_observations_N10 = []

for i in range(len(parameters)):
    k1 = []
    for j in range(1000):
        categorical_observations_10 = X.sample_n(10)
        kk1 = categorical_mle(list(categorical_observations_10))
        k1.append(kk1[i])
    all_observations_N10.append(k1)
    
mean10 = []
stder10 = []
for i in range(len(parameters)):
    mean = np.mean(all_observations_N10[i])
    stder = np.std(all_observations_N10[i])
    mean10.append(mean)
    stder10.append(stder)

## N = 100
all_observations_N100 = []

for i in range(len(parameters)):
    k1 = []
    for j in range(1000):
        categorical_observations_100 = X.sample_n(100)
        kk1 = categorical_mle(list(categorical_observations_100))
        k1.append(kk1[i])
    all_observations_N100.append(k1)
    
mean100 = []
stder100 = []
for i in range(len(parameters)):
    mean = np.mean(all_observations_N100[i])
    stder = np.std(all_observations_N100[i])
    mean100.append(mean)
    stder100.append(stder)

## N = 1000
all_observations_N1000 = []
for i in range(len(parameters)):
    k1 = []
    for j in range(1000):
        categorical_observations_1000 = X.sample_n(1000)
        kk1 = categorical_mle(list(categorical_observations_1000))
        k1.append(kk1[i])
    all_observations_N1000.append(k1)

mean1000 = []
stder1000 = []
for i in range(len(parameters)):
    mean = np.mean(all_observations_N1000[i])
    stder = np.std(all_observations_N1000[i])
    mean1000.append(mean)
    stder1000.append(stder)
    

### Plotting all the means and standard errors for every N
## Means and standard error for N=1
meansn1 = [mean1[0], mean1[1], mean1[2]]
stdern1 = [stder1[0], stder1[1], stder1[2]]

## Means and standard error for N=10
meansn10 = [mean10[0], mean10[1], mean10[2]]
stdern10 = [stder10[0], stder10[1], stder10[2]]

## Means and standard error for N=100
meansn100 = [mean100[0], mean100[1], mean100[2]]
stdern100 = [stder100[0], stder100[1], stder100[2]]

## Means and standard error for N=1000
meansn1000 = [mean1000[0], mean1000[1], mean1000[2]]
stdern1000 = [stder1000[0], stder1000[1], stder1000[2]]

# width of the bars
barWidth = 0.1

# The x position of bars
redbars = np.arange(len(meansn1))
bluebars = [x + barWidth for x in redbars]
greenbars = [x + barWidth for x in bluebars]
yellowbars = [x + barWidth for x in greenbars]

# Create red bars
plt.bar(redbars, meansn1, width = barWidth, color = 'red', yerr=stdern1, capsize=7, label='1')
# Create blue bars
plt.bar(bluebars, meansn10, width = barWidth, color = 'blue', yerr=stdern10, capsize=7, label='10')
# Create green bars
plt.bar(greenbars, meansn100, width = barWidth, color = 'green', yerr=stdern100, capsize=7, label='100')
# Create yellow bars
plt.bar(yellowbars, meansn1000, width = barWidth, color = 'y', yerr=stdern1000, capsize=7, label='1000')

# general layout
plt.xticks([r + barWidth for r in range(len(redbars))], ['1', '2', '3'])
plt.ylabel('MLE')
plt.legend()

plt.show()

