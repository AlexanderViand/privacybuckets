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


# Binomial Noise (scaled)
def binomial_noise(events, n, offset=1, scaling_factor=1):
    n_scaled = n / (scaling_factor ** 2)
    binomial_a = np.zeros(events.size)
    binomial_b = np.zeros(events.size)
    for k in range(0, events.size):
        binomial_a[k] = binom.pmf(events[k] / scaling_factor, n_scaled, 0.5, -n_scaled / 2)
        binomial_b[k] = binom.pmf((events[k] - offset) / scaling_factor, n_scaled, 0.5, -n_scaled / 2)

    binomial_a = binomial_a / np.sum(binomial_a)
    binomial_b = binomial_b / np.sum(binomial_b)
    return binomial_a, binomial_b


# Gaussian noise, truncated
def gaussian_noise(events, scale, offset=1):
    gauss_a = norm.pdf(events, scale=scale)
    gauss_a /= np.sum(gauss_a)

    events_b = map(lambda x: x - offset, events)
    gauss_b = norm.pdf(events_b, scale=scale)
    gauss_b /= np.sum(gauss_b)

    return gauss_a, gauss_b


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


def plot_multiple(events_and_curve, labels, title='', xlabel='', ylabel='', xlim=None, ylim=None):
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
    offset = 1
    compositions = 640
    eps_vector = np.linspace(0, 3, 100)
    truncation_at = 2500
    events = np.arange(-truncation_at, truncation_at + 1, 1)

    # Define distributions
    bin_a, bin_b = binomial_noise(events, n / 16, offset=offset, scaling_factor=1)
    bin_scaled_a, bin_scaled_b = binomial_noise(events, n, offset=offset, scaling_factor=4)

    # Show input distributions
    input_curves = (bin_a, bin_b, bin_scaled_a, bin_scaled_b)
    input_labels = ("Binomial A", "Binomial B", "Binomial (scaled) A", "Binomial (scaled) B"
                    )
    plot(events, input_curves, input_labels, title="Input distributions", xlabel="Noise", ylabel="mass")

    # Show input distributions restricted to interesting part
    plot(events, input_curves, input_labels, title="Input distributions", xlabel="Noise", ylabel="mass",
         xlim=(-2 * scale, 2 * scale))

    # Run Privacy Buckets
    composed_bin = compose(bin_a, bin_b, compositions=compositions)
    composed_bin_scaled = compose(bin_scaled_a, bin_scaled_b, compositions=compositions)

    # abusing internals, we can look at the bucket distribution
    composed_curves = (composed_bin.bucket_distribution, composed_bin_scaled.bucket_distribution)
    composed_labels = ('bucket distribution Binomial', 'bucket distribution Binomial (scaled)')
    plot_multiple(composed_curves, composed_labels, "Bucket Distributions", "bucket number", "mass")

    # Now we build the delta(eps) graphs from the computed distribution
    upper_bound_bin = [composed_bin.delta_of_eps_upper_bound(eps) for eps in eps_vector]
    lower_bound_bin = [composed_bin.delta_of_eps_lower_bound(eps) for eps in eps_vector]
    upper_bound_bin_scaled = [composed_bin_scaled.delta_of_eps_upper_bound(eps) for eps in eps_vector]
    lower_bound_bin_scaled = [composed_bin_scaled.delta_of_eps_lower_bound(eps) for eps in eps_vector]

    # Show all curves
    delta_eps_curves = (upper_bound_bin, lower_bound_bin, upper_bound_bin_scaled, lower_bound_bin_scaled)
    labels = ("Bin (u)", "Bin (l)", "Bin scaled (u)", "Bin scaled (l)")
    plot(eps_vector, delta_eps_curves, labels)  # , xlim=(0.5, 2.0), ylim=(0, 10 ** -5))

    # Save to csv so we can do the plotting separately
    # np.savetxt("delta-eps_approximations.csv",
    #            np.transpose([eps_vector, upper_bound_gauss, upper_bound_bin2, upper_bound_bin3, upper_bound_bin4]),
    #            delimiter=',')


if __name__ == "__main__":
    main()
