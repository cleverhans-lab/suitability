# Description: Statistical tests for suitability filter
from scipy import stats
import numpy as np


def t_test(sample1, sample2, equal_var=False):
    '''
    Perform a two-sample t-test for two samples.
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    equal_var: if False, uses Welch's t-test.
    Returns: t-statistic, p-value
    '''
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
    return t_stat, p_value


def non_inferiority_test(sample1, sample2, threshold=0, equal_var=False, increase_good=True):
    '''
    Perform a non-inferiority test for two samples.
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    threshold: non-inferiority threshold (e.g., 0.05 for 5%, relative to mean of sample1)
    equal_var: if False, uses Welch's t-test.
    increase_good: if True, Ho: mean2 <= mean1 - threshold. Else Ho: mean2 >= mean1 + threshold.
    Returns: t-statistic, p-value
    '''
    if threshold != 0:
        difference = threshold * np.mean(sample1)
        if increase_good:
            sample2 += difference
        else:
            sample2 -= difference

    # Perform two-sided t-test
    t_stat, p_value_two_sided = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
    
    # Adjust for one-sided test (upper-tailed)
    if increase_good:
        p_value_one_sided = p_value_two_sided / 2
    else:
        p_value_one_sided = 1 - (p_value_two_sided / 2)
    
    return t_stat, p_value_one_sided


def satterthwaite_dof(s1, n1, s2, n2):
    """Calculate the Satterthwaite degrees of freedom."""
    numerator = (s1**2/n1 + s2**2/n2)**2
    denominator = ((s1**2/n1)**2 / (n1 - 1)) + ((s2**2/n2)**2 / (n2 - 1))
    return numerator / denominator


def equivalence_test(sample1, sample2, threshold_low, threshold_upp, equal_var=False):
    """
    Perform a corrected custom TOST with Satterthwaite's degrees of freedom.
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    threshold_low: lower bound of the equivalence interval as a fraction of the mean of sample1.
    threshold_upp: upper bound of the equivalence interval as a fraction of the mean of sample1.
    equal_var: if False, uses Welch's t-test.
    Returns: t-statistic and p-values for the lower and upper bound tests, and the degrees of freedom.
    """

    # Calculate means and standard deviations
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    n1, n2 = len(sample1), len(sample2)
    mean_diff = mean1 - mean2
    low = mean1 * threshold_low
    upp = mean1 * threshold_upp

    # Calculate standard error of the difference
    se_diff = np.sqrt(std1**2/n1 + std2**2/n2)

    # Satterthwaite's degrees of freedom
    dof = satterthwaite_dof(std1, n1, std2, n2)

    # Lower bound test
    t_stat_low = (mean_diff - low) / se_diff
    p_value_low = 1 - stats.t.cdf(t_stat_low, df=dof)

    # Upper bound test
    t_stat_upp = (mean_diff - upp) / se_diff
    p_value_upp = 1 - stats.t.cdf(-t_stat_upp, df=dof)


    # Return the results
    return t_stat_low, p_value_low, t_stat_upp, p_value_upp, dof



