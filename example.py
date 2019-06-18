import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom, norm
from core.probabilitybuckets_light import ProbabilityBuckets

####
# Privacy Buckets is based on the following publication
####

# [*] S. Meiser, E. Mohammadi, 
# [*] "Tight on Budget? Tight Bounds for r-Fold Approximate Differential Privacy", 
# [*] Proceedings of the 25th ACM Conference on Computer and Communications Security (CCS), 2018


# Parameters
scale = 20
n = 4 * (scale ** 2)
compositions = 2 ** 7

# Derived & Helper Variables
truncation_at = 2500
number_of_events = int(2 * truncation_at + 1)
events = np.linspace(-truncation_at, truncation_at, number_of_events, endpoint=True)
x_axis = np.linspace(-truncation_at, truncation_at + 1, number_of_events + 1, endpoint=True)

# Binomial Noise
binA = np.zeros(x_axis.size)
binB = np.zeros(x_axis.size)
for k in range(0, x_axis.size):
    binA[k] = binom.pmf(x_axis[k], n, 0.5, -n/2)
    binB[k] = binom.pmf(x_axis[k-1], n, 0.5, -n/2)


# Clamped Binomial noise
# truncatedBinA = np.zeros(n + 1)
# truncatedBinB = np.zeros(n + 1)
# truncatedBinA[0] = binom.pmf(0, n, 0.5) + binom.pmf(1, n, 0.5)
# truncatedBinB[0] = binom.pmf(0, n, 0.5)
# truncatedBinA[n] = binom.pmf(n, n, 0.5)
# truncatedBinB[n] = binom.pmf(n - 1, n, 0.5) + binom.pmf(n, n, 0.5)
# for k in range(1, n - 1):
#     truncatedBinA[k] = binom.pmf(k, n, 0.5)
#     truncatedBinB[k] = binom.pmf(k - 1, n, 0.5)
# truncatedBinA /= np.sum(truncatedBinA)
# truncatedBinB /= np.sum(truncatedBinB)

# Gaussian noise, truncated
distribution = norm.pdf(events, scale=scale)
distribution /= np.sum(distribution)

gaussA = np.zeros(x_axis.size)
gaussA[:number_of_events] = distribution

gaussB = np.zeros(x_axis.size)
gaussB[1:] = distribution

# Show input distributions
plt.plot(x_axis, gaussA, label='Gauss A')
plt.plot(x_axis, gaussB, label='Gauss B')
plt.plot(x_axis, binA, label='Binomial A')
plt.plot(x_axis, binB, label='Binomial B')
# plt.plot(x_axis, truncatedBinA, label='Truncated Binomial A')
# plt.plot(x_axis, truncatedBinB, label='Truncated Binomial B')

plt.title("Input distributions")
plt.xlabel("Noise")
plt.ylabel("mass")
plt.legend()
plt.show()

# # Initialize privacy buckets for gauss.
privacybucketsG = ProbabilityBuckets(
        number_of_buckets=100000,   # number of buckets. The more the better as the resolution gets more finegraind
        factor = 1 + 1e-4,   # depends on the number_of_buckets and is the multiplicative constant between two buckets.
        dist1_array = gaussA,  # distribution A
        dist2_array = gaussB,  # distribution B
        caching_directory = "./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10**(-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
        )

# Now we evaluate how the distributon looks after 2**k independent compositions
# input can be arbitrary positive integer, but exponents of 2 are numerically the most stable
privacybuckets_composedG = privacybucketsG.compose(compositions)

# Print status summary
privacybuckets_composedG.print_state()

# # Initialize privacy buckets for binomial.
privacybucketsB = ProbabilityBuckets(
        number_of_buckets=100000,   # number of buckets. The more the better as the resolution gets more finegraind
        factor = 1 + 1e-4,   # depends on the number_of_buckets and is the multiplicative constant between two buckets.
        dist1_array =  binA,  # distribution A
        dist2_array = binB,  # distribution B
        caching_directory = "./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10**(-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
        )

# Now we evaluate how the distributon looks after 2**k independent compositions
# input can be arbitrary positive integer, but exponents of 2 are numerically the most stable
privacybuckets_composedB = privacybucketsB.compose(compositions)

# Print status summary
privacybuckets_composedB.print_state()

# abusing internals, we can look at the bucket distribution
plt.plot(privacybuckets_composedG.bucket_distribution)
plt.title("bucket distribution Gauss")
plt.xlabel("bucket number")
plt.ylabel("mass")
plt.show()
# abusing internals, we can look at the bucket distribution
plt.plot(privacybuckets_composedB.bucket_distribution)
plt.title("bucket distribution Binomial")
plt.xlabel("bucket number")
plt.ylabel("mass")
plt.show()

# Now we build the delta(eps) graphs from the computed distribution
# Gauss
eps_vector =  np.linspace(0,3,100)
upper_boundG = [privacybuckets_composedG.delta_of_eps_upper_bound(eps) for eps in eps_vector]
lower_boundG = [privacybuckets_composedG.delta_of_eps_lower_bound(eps) for eps in eps_vector]
# Binomial
upper_boundB = [privacybuckets_composedB.delta_of_eps_upper_bound(eps) for eps in eps_vector]
lower_boundB = [privacybuckets_composedB.delta_of_eps_lower_bound(eps) for eps in eps_vector]

plt.plot(eps_vector, upper_boundG, label="upper_bound G")
plt.plot(eps_vector, lower_boundG, label="lower_bound G")
plt.plot(eps_vector, upper_boundB, label="upper_bound B")
plt.plot(eps_vector, lower_boundB, label="lower_bound B")
plt.legend()
plt.title("Mechanisms with scale sigma ={:d} (or n={:d}) after {:d} compositions".format(scale, n, compositions))
plt.xlabel("eps")
plt.ylabel("delta")
plt.ticklabel_format(useOffset=False)  # Hotfix for the behaviour of my current matplotlib version
plt.show()



