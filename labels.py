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
scale = 40
n1 = 4 * (scale ** 2)

compositions = 283

# Derived & Helper Variables
approx_factor = 2
n2 =  int((4.0/(approx_factor**2)) * (scale ** 2))
n = n2
truncation_at = 2500
number_of_events = int(2 * truncation_at + 1)
events = np.arange(-truncation_at, truncation_at + 1, 1)
x_axis = np.arange(-truncation_at, truncation_at + 2, 1)

# Binomial Noise (scaled)
binA2 = np.zeros(x_axis.size)
binB2 = np.zeros(x_axis.size)
for k in range(0, x_axis.size):
    binA2[k] = binom.pmf(x_axis[k]/approx_factor, n, 0.5, -n/2)
    binB2[k] = binom.pmf(x_axis[(k-1)]/approx_factor, n, 0.5, -n/2)

binA2 = binA2/np.sum(binA2)
binB2 = binB2/np.sum(binB2)


# Derived & Helper Variables
approx_factor = 3
n3 =  int((4.0/(approx_factor**2)) * (scale ** 2))
n = n3
truncation_at = 2500
number_of_events = int(2 * truncation_at + 1)
events = np.arange(-truncation_at, truncation_at + 1, 1)
x_axis = np.arange(-truncation_at, truncation_at + 2, 1)

# Binomial Noise (scaled)
binA3 = np.zeros(x_axis.size)
binB3 = np.zeros(x_axis.size)
for k in range(0, x_axis.size):
    binA3[k] = binom.pmf(x_axis[k]/approx_factor, n, 0.5, -n/2)
    binB3[k] = binom.pmf(x_axis[(k-1)]/approx_factor, n, 0.5, -n/2)

binA3 = binA3/np.sum(binA3)
binB3 = binB3/np.sum(binB3)


# Derived & Helper Variables
approx_factor = 4
n4 =  int((4.0/(approx_factor**2)) * (scale ** 2))
n = n4
truncation_at = 2500
number_of_events = int(2 * truncation_at + 1)
events = np.arange(-truncation_at, truncation_at + 1, 1)
x_axis = np.arange(-truncation_at, truncation_at + 2, 1)

# Binomial Noise (scaled)
binA4 = np.zeros(x_axis.size)
binB4 = np.zeros(x_axis.size)
for k in range(0, x_axis.size):
    binA4[k] = binom.pmf(x_axis[k]/approx_factor, n, 0.5, -n/2)
    binB4[k] = binom.pmf(x_axis[(k-1)]/approx_factor, n, 0.5, -n/2)

binA4 = binA4/np.sum(binA4)
binB4 = binB4/np.sum(binB4)


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
plt.plot(x_axis, binA2, label='Binomial A (f=2)')
plt.plot(x_axis, binB2, label='Binomial B (f=2)')
plt.plot(x_axis, binA3, label='Binomial A (f=3)')
plt.plot(x_axis, binB3, label='Binomial B (f=3)')
plt.plot(x_axis, binA4, label='Binomial A (f=4)')
plt.plot(x_axis, binB4, label='Binomial B (f=4)')
# plt.plot(x_axis, truncatedBinA, label='Truncated Binomial A')
# plt.plot(x_axis, truncatedBinB, label='Truncated Binomial B')

plt.title("Input distributions")
plt.xlabel("Noise")
plt.ylabel("mass")
plt.legend()
plt.show()

# Show input distributions restricted to interesting part
plt.plot(x_axis, gaussA, label='Gauss A')
plt.plot(x_axis, gaussB, label='Gauss B')
plt.plot(x_axis, binA2, label='Binomial A (f=2)')
plt.plot(x_axis, binB2, label='Binomial B (f=2)')
plt.plot(x_axis, binA3, label='Binomial A (f=3)')
plt.plot(x_axis, binB3, label='Binomial B (f=3)')
plt.plot(x_axis, binA4, label='Binomial A (f=4)')
plt.plot(x_axis, binB4, label='Binomial B (f=4)')
# plt.plot(x_axis, truncatedBinA, label='Truncated Binomial A')
# plt.plot(x_axis, truncatedBinB, label='Truncated Binomial B')

plt.title("Input distributions (near mean)")
plt.xlabel("Noise")
plt.ylabel("mass")
plt.xlim(-2*scale, 2*scale)
plt.legend()
plt.show()

# # Initialize privacy buckets for gauss.
privacybucketsG = ProbabilityBuckets(
        number_of_buckets=100000,   # number of buckets. The more the better as the resolution gets more finegraind
        factor = 1 + 1e-4,   # depends on the number_of_buckets and is the multiplicative constant between two buckets.
        dist1_array = gaussA,  # distribution A
        dist2_array = gaussB,  # distribution B
        caching_directory = "./pb-cacheGauss",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10**(-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
        )

# Now we evaluate how the distributon looks after 2**k independent compositions
# input can be arbitrary positive integer, but exponents of 2 are numerically the most stable
privacybuckets_composedG = privacybucketsG.compose(compositions)

# Print status summary
privacybuckets_composedG.print_state()

# # Initialize privacy buckets for binomial.
privacybucketsB2 = ProbabilityBuckets(
        number_of_buckets=100000,   # number of buckets. The more the better as the resolution gets more finegraind
        factor = 1 + 1e-4,   # depends on the number_of_buckets and is the multiplicative constant between two buckets.
        dist1_array =  binA2,  # distribution A
        dist2_array = binB2,  # distribution B
        caching_directory = "./pb-cacheBinom",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10**(-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
        )

# Now we evaluate how the distributon looks after 2**k independent compositions
# input can be arbitrary positive integer, but exponents of 2 are numerically the most stable
privacybuckets_composedB2 = privacybucketsB2.compose(compositions)

# Print status summary
privacybuckets_composedB2.print_state()


# # Initialize privacy buckets for binomial.
privacybucketsB3 = ProbabilityBuckets(
        number_of_buckets=100000,   # number of buckets. The more the better as the resolution gets more finegraind
        factor = 1 + 1e-4,   # depends on the number_of_buckets and is the multiplicative constant between two buckets.
        dist1_array =  binA3,  # distribution A
        dist2_array = binB3,  # distribution B
        caching_directory = "./pb-cacheBinom",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10**(-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
        )

# Now we evaluate how the distributon looks after 2**k independent compositions
# input can be arbitrary positive integer, but exponents of 2 are numerically the most stable
privacybuckets_composedB3 = privacybucketsB3.compose(compositions)

# Print status summary
privacybuckets_composedB3.print_state()

# # Initialize privacy buckets for binomial.
privacybucketsB4 = ProbabilityBuckets(
        number_of_buckets=100000,   # number of buckets. The more the better as the resolution gets more finegraind
        factor = 1 + 1e-4,   # depends on the number_of_buckets and is the multiplicative constant between two buckets.
        dist1_array =  binA4,  # distribution A
        dist2_array = binB4,  # distribution B
        caching_directory = "./pb-cacheBinom",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10**(-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
        )

# Now we evaluate how the distributon looks after 2**k independent compositions
# input can be arbitrary positive integer, but exponents of 2 are numerically the most stable
privacybuckets_composedB4 = privacybucketsB4.compose(compositions)

# Print status summary
privacybuckets_composedB4.print_state()


# abusing internals, we can look at the bucket distribution
plt.plot(privacybuckets_composedG.bucket_distribution)
plt.title("bucket distribution Gauss")
plt.xlabel("bucket number")
plt.ylabel("mass")
plt.show()
# abusing internals, we can look at the bucket distribution
plt.plot(privacybuckets_composedB2.bucket_distribution)
plt.title("bucket distribution Binomial (f=2)")
plt.xlabel("bucket number")
plt.ylabel("mass")
plt.show()
# abusing internals, we can look at the bucket distribution
plt.plot(privacybuckets_composedB3.bucket_distribution)
plt.title("bucket distribution Binomial (f=4)")
plt.xlabel("bucket number")
plt.ylabel("mass")
plt.show()
# abusing internals, we can look at the bucket distribution
plt.plot(privacybuckets_composedB4.bucket_distribution)
plt.title("bucket distribution Binomial (f=4)")
plt.xlabel("bucket number")
plt.ylabel("mass")
plt.show()

# Now we build the delta(eps) graphs from the computed distribution
# Gauss
eps_vector =  np.linspace(0,3,100)
upper_boundG = [privacybuckets_composedG.delta_of_eps_upper_bound(eps) for eps in eps_vector]
lower_boundG = [privacybuckets_composedG.delta_of_eps_lower_bound(eps) for eps in eps_vector]
# Binomial
upper_boundB2 = [privacybuckets_composedB2.delta_of_eps_upper_bound(eps) for eps in eps_vector]
lower_boundB2 = [privacybuckets_composedB2.delta_of_eps_lower_bound(eps) for eps in eps_vector]
upper_boundB3 = [privacybuckets_composedB3.delta_of_eps_upper_bound(eps) for eps in eps_vector]
lower_boundB3 = [privacybuckets_composedB3.delta_of_eps_lower_bound(eps) for eps in eps_vector]
upper_boundB4 = [privacybuckets_composedB4.delta_of_eps_upper_bound(eps) for eps in eps_vector]
lower_boundB4 = [privacybuckets_composedB4.delta_of_eps_lower_bound(eps) for eps in eps_vector]

delta_point = 10**-5
plt.plot(eps_vector, upper_boundG, '-b', label="GaussianNoise (or f=1, n={:d})".format(n1))
pointG = np.abs(np.array(upper_boundG)-delta_point).argmin()
plt.text(eps_vector[pointG], 1.1*delta_point, "e ={:f}".format(eps_vector[pointG]))
#plt.plot(eps_vector, lower_boundG, label="lower_bound G")

plt.plot(eps_vector, upper_boundB2, '--g', label="Binomial Noise (f=2, n={:d})".format(n2))
pointB2 = np.abs(np.array(upper_boundB2)-delta_point).argmin()
plt.text(eps_vector[pointB2], 0.9*delta_point, "e ={:f}".format(eps_vector[pointB2]))

plt.plot(eps_vector, upper_boundB3, '--r', label="Binomial Noise (f=3, n={:d})".format(n3))
pointB3 = np.abs(np.array(upper_boundB3)-delta_point).argmin()
plt.text(eps_vector[pointB3], 0.8*delta_point, "e ={:f}".format(eps_vector[pointB3]))

plt.plot(eps_vector, upper_boundB4, '--y', label="Binomial Noise (f=4, n={:d})".format(n4))
pointB4 = np.abs(np.array(upper_boundB4)-delta_point).argmin()
plt.text(eps_vector[pointB4], 0.7*delta_point, "e ={:f}".format(eps_vector[pointB4]))

#plt.plot(eps_vector, lower_boundB, label="lower_bound B")
plt.legend()
plt.title("Mechanisms with scale sigma ={:d} after {:d} compositions".format(scale, compositions))
plt.xlabel("eps")
plt.ylabel("delta")
plt.axes().axhline(y=delta_point)
plt.ylim(0,2*10**-5)
plt.xlim(0.5,2.5)
# plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=False, useMathText=True)
plt.ticklabel_format(useOffset=False)  # Hotfix for the behaviour of my current matplotlib version
plt.show()



