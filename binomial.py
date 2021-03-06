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


# Gaussian noise, truncated
def gaussian_noise(events, scale, offset=1):
    gauss_a = norm.pdf(events, scale=scale)
    gauss_a /= np.sum(gauss_a)

    events_b = list(map(lambda x: x - offset, events))
    gauss_b = norm.pdf(events_b, scale=scale)
    gauss_b /= np.sum(gauss_b)

    return gauss_a, gauss_b


# Run Privacy Buckets
def compose(distribution_a, distribution_b, compositions=1):
    privacybuckets = ProbabilityBuckets(
        number_of_buckets=300000,  # number of buckets. The more the better as the resolution gets more finegraind
        factor=1 + 1e-3,  # depends on the number_of_buckets and is the multiplicative constant between two buckets.
        dist1_array=distribution_a,  # distribution A
        dist2_array=distribution_b,  # distribution B
        caching_directory="./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
        free_infty_budget=10 ** (-20),  # how much we can put in the infty bucket before first squaring
        error_correction=True,  # error correction. See publication for details
    )

    privacybuckets.print_state()
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

def plot_combined(events_and_curves, labels, title='', xlabel='', ylabel='', xlim=None, ylim=None):
    mpl.rcParams['figure.dpi'] = 300
    b = True
    plt.plot(events_and_curves, label=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend()
    plt.show()


def plot_multiple(events_and_curve, labels, title='', xlabel='', ylabel='', xlim=None, ylim=None):
    mpl.rcParams['figure.dpi'] = 300
    for p, l in zip(events_and_curve, labels):
        plt.plot(p, label=l)
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
    compositions = 2
    eps_vector = np.linspace(0, 3, 100)
    truncation_at = 2500
    events = np.arange(-truncation_at, truncation_at + 1, 1)

    # Define distributions
    bin3_a, bin3_b = binomial_noise(events, n / 9, offset=offset)

    # Show input distributions
    input_curves = (bin3_a, bin3_b)
    input_labels = (
        'Binomial (n={}) A'.format(n / 9),
        'Binomial (n={}) B'.format(n / 9)
    )
    plot(events, input_curves, input_labels, title="Input distributions", xlabel="Noise", ylabel="mass")

    # Show input distributions restricted to interesting part
    plot(events, input_curves, input_labels, title="Input distributions", xlabel="Noise", ylabel="mass",
         xlim=(-2 * scale, 2 * scale))

    # Run Privacy Buckets
    composed_bin3 = compose(bin3_a, bin3_b, compositions=compositions)

    # abusing internals, we can look at the bucket distribution
    plot_combined(composed_bin3.bucket_distribution, 'bucket dist')

    # Now we build the delta(eps) graphs from the computed distribution
    upper_bound_bin3 = [composed_bin3.delta_of_eps_upper_bound(eps) for eps in eps_vector]
    lower_bound_bin3 = [composed_bin3.delta_of_eps_lower_bound(eps) for eps in eps_vector]

    # Show all curves
    delta_eps_curves = (

        upper_bound_bin3, lower_bound_bin3
    )
    labels = (
              'Binomial (n={}) upper'.format(n / 9), 'Binomial (n={}) lower'.format(n / 9)
    )
    plot(eps_vector, delta_eps_curves, labels)

    # Save to csv so we can do the plotting separately
    # np.savetxt("delta-eps.csv",
    #            np.transpose([eps_vector, upper_bound_gauss, upper_bound_bin2, upper_bound_bin3, upper_bound_bin4]),
    #            delimiter=',')


if __name__ == "__main__":
    main()
