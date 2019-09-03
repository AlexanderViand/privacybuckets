import numpy as np
import matplotlib as mpl

mpl.use('Qt5Agg')
from matplotlib import pyplot as plt
from scipy.stats import binom, norm

from core.probabilitybuckets_light import ProbabilityBuckets


####
# Privacy Buckets is based on the following publication
####

# [*] S. Meiser, E. Mohammadi, 
# [*] "Tight on Budget? Tight Bounds for r-Fold Approximate Differential Privacy", 
# [*] Proceedings of the 25th ACM Conference on Computer and Communications Security (CCS), 2018


# Binomial Noise (scaled)
def binomial_noise(events, n, offset=1):
    binomial_a = np.zeros(events.size)
    binomial_b = np.zeros(events.size)
    for k in range(0, events.size):
        binomial_a[k] = binom.pmf(events[k], n, 0.5, -n / 2)
        binomial_b[k] = binom.pmf(events[k] - offset, n, 0.5, -n / 2)
    # Should already be normalized, but just to make sure
    #binomial_a = binomial_a / np.sum(binomial_a)
    #binomial_b = binomial_b / np.sum(binomial_b)
    return binomial_a, binomial_b

# Buckets for binomial noise
def binomial_buckets(n, factor, number_of_buckets): #offset=1 for now
    #non-distinguishing events
    l_ab = np.zeros(n+2)
    for k in range(1, n):
        l_ab[k] = np.log(( n - k + 1) / k)
    # for 0 it's infinity
    # for n+1 it's 0
    log_factor = np.log(factor, dtype=np.float64)
    indices = l_ab / log_factor + number_of_buckets // 2
    indices = np.ceil(indices).astype(int)

    # # fill buckets
    # self.infty_bucket = np.float64(0.0)
    # self.distinguishing_events = np.float64(0.0)
    # for i, m_infty, m_null, a, err in zip(indices, infty_mask, null_mask, distr1, errors):
    #     if
    # m_infty:
    # self.distinguishing_events += a
    # # self.infty_bucket += a
    # continue
    # if m_null:
    #     continue
    # # i = int(np.ceil(i))
    # if i >= self.number_of_buckets:
    #     self.infty_bucket += a
    #     continue
    # if i < 0:
    #     self.bucket_distribution[0] += a
    #     virtual_error[0] += err
    #     continue
    # self.bucket_distribution[i] += a
    # virtual_error[i] += err


    # self.one_index = int(self.number_of_buckets // 2)
    # self.u = np.int64(1)




# Run Privacy Buckets
def compose(distribution_a, distribution_b, compositions=1):
    privacybuckets = ProbabilityBuckets(
        number_of_buckets=100000,  # number of buckets. The more the better as the resolution gets more finegraind
        factor=1 + 1e-5,  # depends on the number_of_buckets and is the multiplicative constant between two buckets.
        dist1_array=distribution_a,  # distribution A
        dist2_array=distribution_b,  # distribution B
        caching_directory="./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10 ** (-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
    )

    # Now we evaluate how the distributon looks after 2**k independent compositions
    # input can be arbitrary positive integer, but exponents of 2 are numerically the most stable
    composed = privacybuckets.compose(compositions)
    composed.print_state()
    return composed


def plot(events, curves, labels, title='', xlabel='', ylabel='', xlim=None, ylim=None):
    mpl.rcParams['figure.dpi'] = 300
    b = True
    for c, l in zip(curves, labels):
        b = not b
        if b:
            plt.plot(events, c, label=l, linestyle="--")
        else:
            plt.plot(events, c, label=l)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.show()


def main():
    # Parameters
    scale = 150
    n = 4 * (scale ** 2)  # approximate Gaussian
    offset = 2
    compositions = 640
    eps_vector = np.linspace(0, 3, 100)
    truncation_at = 2500
    events = np.arange(-truncation_at, truncation_at + 1, 1)

    # Define distributions
    bin_a, bin_b = binomial_noise(events, n, offset=offset)

    # Show input distribution
    input_curves = (bin_a, bin_b)
    input_labels = ('Binomial (n={}) A'.format(n), 'Binomial (n={}) B'.format(n))
    plot(events, input_curves, input_labels, title="Input distributions", xlabel="Noise", ylabel="mass")

    # Show input distributions restricted to interesting part
    plot(events, input_curves, input_labels, title="Input distributions", xlabel="Noise", ylabel="mass",
         xlim=(-2 * scale, 2 * scale))

    # Run Privacy Buckets
    composed_bin = compose(bin_a, bin_b, compositions=compositions)


    # abusing internals, we can look at the bucket distribution
    plot(composed_bin.bucket_distribution, 'bucket distribution Binomial (n={})'.format(n), "Bucket Distributions", "bucket number", "mass")

    # Now we build the delta(eps) graphs from the computed distribution
    upper_bound_bin = [composed_bin.delta_of_eps_upper_bound(eps) for eps in eps_vector]
    # lower_bound_bin = [composed_bin.delta_of_eps_lower_bound(eps) for eps in eps_vector]


    labels = ('Binomial (n={}) upper'.format(n))
    plot(eps_vector, upper_bound_bin, labels, xlim=(0.5, 10.0), ylim=(0, 10 ** -5))

    # Save to csv so we can do the plotting separately
    np.savetxt("delta-eps-custom-buckets.csv",
               np.transpose([eps_vector, upper_bound_bin]),
               delimiter=',')


if __name__ == "__main__":
    main()
