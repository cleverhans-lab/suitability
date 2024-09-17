# Description: Statistical tests for suitability filter
import numpy as np
from scipy import stats


def t_test(sample1, sample2, equal_var=False):
    """
    Perform a two-sample t-test for two samples.
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    equal_var: if False, uses Welch's t-test.
    Returns: t-statistic, p-value
    """
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
    return {"t_statistic": t_stat, "p_value": p_value}


def non_inferiority_ztest(array1, array2, margin=0, increase_good=True, alpha=0.05):
    """
    Perform a non-inferiority z-test for two arrays.
    Use when: large sample size, approximately normal data distribution, assumes known population variances
    array1: array of values for sample 1
    array2: array of values for sample 2
    margin: non-inferiority margin (threshold for difference in means)
    increase_good: if True, Ho: mean2 <= mean1 - threshold. Else Ho: mean2 >= mean1 + threshold.
    alpha: significance level
    Returns: mean_diff, z_score, p_value, reject_null
    """

    # Calculate the mean and standard deviation of both arrays
    mean1 = np.mean(array1)
    mean2 = np.mean(array2)
    std1 = np.std(array1, ddof=1)
    std2 = np.std(array2, ddof=1)

    # Calculate the difference in means
    if increase_good:
        mean_diff = mean1 - mean2
    else:
        mean_diff = mean2 - mean1

    # Calculate the standard error of the difference
    se_diff = np.sqrt((std1**2 / len(array1)) + (std2**2 / len(array2)))

    # Calculate the Z-score
    z_score = (mean_diff - margin) / se_diff

    # Calculate the p-value
    p_value = stats.norm.cdf(z_score)

    return {
        "mean_diff": mean_diff,
        "z_score": z_score,
        "p_value": p_value,
        "reject_null": p_value < alpha,
    }


def non_inferiority_ttest(
    sample1, sample2, margin=0, increase_good=True, equal_var=False, alpha=0.05
):
    """
    Perform a non-inferiority t-test for two samples.
    Use when: small sample size, unequal population variances, adjusts for dof, accounts for sample size differences
    sample1: array of values for sample 1 (typically validation data provided by model provider)
    sample2: array of values for sample 2 (typically sample provided by model user)
    margin: non-inferiority margin (threshold for difference in means)
    equal_var: if False, uses Welch's t-test.
    increase_good: if True, Ho: mean2 <= mean1 - threshold. Else Ho: mean2 >= mean1 + threshold.
    Returns: t_statistic, p_value, reject_null
    """
    if increase_good:
        sample2_diff = sample2 + margin
    else:
        sample2_diff = sample2 - margin

    # Perform two-sided t-test
    t_stat, p_value_two_sided = stats.ttest_ind(
        sample1, sample2_diff, equal_var=equal_var
    )

    is_neg = t_stat < 0

    # Adjust for one-sided test (upper-tailed)
    if increase_good and is_neg or not increase_good and not is_neg:
        p_value_one_sided = p_value_two_sided / 2
    else:
        p_value_one_sided = 1 - (p_value_two_sided / 2)

    return {
        "t_statistic": t_stat,
        "p_value": p_value_one_sided,
        "reject_null": p_value_one_sided < alpha,
    }


def satterthwaite_dof(s1, n1, s2, n2):
    """Calculate the Satterthwaite degrees of freedom."""
    numerator = (s1**2 / n1 + s2**2 / n2) ** 2
    denominator = ((s1**2 / n1) ** 2 / (n1 - 1)) + ((s2**2 / n2) ** 2 / (n2 - 1))
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
    se_diff = np.sqrt(std1**2 / n1 + std2**2 / n2)

    # Satterthwaite's degrees of freedom
    dof = satterthwaite_dof(std1, n1, std2, n2)

    # Lower bound test
    t_stat_low = (mean_diff - low) / se_diff
    p_value_low = 1 - stats.t.cdf(t_stat_low, df=dof)

    # Upper bound test
    t_stat_upp = (mean_diff - upp) / se_diff
    p_value_upp = 1 - stats.t.cdf(-t_stat_upp, df=dof)

    # Return the results
    return {
        "t_statistic_low": t_stat_low,
        "p_value_low": p_value_low,
        "t_statistic_upp": t_stat_upp,
        "p_value_upp": p_value_upp,
        "dof": dof,
    }
