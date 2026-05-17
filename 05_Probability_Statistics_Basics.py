"""
Probability & Statistics Basics for Machine Learning
Practical Python examples using NumPy
Essential concepts for understanding ML algorithms
"""

import numpy as np
from collections import Counter


def section_divider(title):
    """Print a formatted section divider"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_basic_probability():
    """Demonstrate basic probability concepts"""
    section_divider("1. BASIC PROBABILITY")
    
    print("Probability: Measure of likelihood (0 to 1)")
    print("  • P(Event) = 0 → Impossible")
    print("  • P(Event) = 1 → Certain")
    print("  • P(Event) = 0.5 → 50% chance\n")
    
    # Coin flip simulation
    print("--- Coin Flip Example ---")
    np.random.seed(42)
    n_flips = 1000
    flips = np.random.choice(['H', 'T'], size=n_flips)
    
    heads_count = np.sum(flips == 'H')
    prob_heads = heads_count / n_flips
    
    print(f"Number of flips: {n_flips}")
    print(f"Heads: {heads_count}")
    print(f"Tails: {n_flips - heads_count}")
    print(f"P(Heads) = {prob_heads:.3f}")
    print(f"P(Tails) = {1 - prob_heads:.3f}")
    print(f"Expected: 0.5 for fair coin")
    
    # Dice roll
    print("\n--- Dice Roll Example ---")
    n_rolls = 10000
    rolls = np.random.randint(1, 7, size=n_rolls)
    
    print(f"Number of rolls: {n_rolls}")
    for i in range(1, 7):
        count = np.sum(rolls == i)
        prob = count / n_rolls
        print(f"P({i}) = {prob:.3f} (Count: {count})")
    print(f"Expected: ~0.167 for fair die")
    
    # Basic probability rules
    print("\n--- Probability Rules ---")
    print("1. Sum Rule: P(A or B) = P(A) + P(B) - P(A and B)")
    print("2. Product Rule (independent): P(A and B) = P(A) × P(B)")
    print("3. Complement: P(not A) = 1 - P(A)")
    print("4. Total Probability: Sum of all outcomes = 1")


def demo_conditional_probability():
    """Demonstrate conditional probability"""
    section_divider("2. CONDITIONAL PROBABILITY")
    
    print("Conditional Probability: P(A|B) = P(A given B)")
    print("Formula: P(A|B) = P(A and B) / P(B)\n")
    
    # Example: Weather and commute
    print("--- Example: Weather and Traffic ---")
    print("\nSimulated data (1000 days):\n")
    
    np.random.seed(42)
    n_days = 1000
    
    # Simulate: 30% rainy days
    rainy = np.random.random(n_days) < 0.3
    
    # If rainy, 70% chance of traffic; if sunny, 20% chance
    traffic = np.where(rainy, 
                       np.random.random(n_days) < 0.7,
                       np.random.random(n_days) < 0.2)
    
    # Calculate probabilities
    p_rain = np.mean(rainy)
    p_traffic = np.mean(traffic)
    p_rain_and_traffic = np.mean(rainy & traffic)
    p_traffic_given_rain = p_rain_and_traffic / p_rain if p_rain > 0 else 0
    
    print(f"P(Rain) = {p_rain:.3f}")
    print(f"P(Traffic) = {p_traffic:.3f}")
    print(f"P(Rain AND Traffic) = {p_rain_and_traffic:.3f}")
    print(f"P(Traffic | Rain) = {p_traffic_given_rain:.3f}")
    print("\nInterpretation: Given it's raining, ~70% chance of traffic")
    
    # Bayes' Theorem
    print("\n--- Bayes' Theorem ---")
    print("Formula: P(A|B) = P(B|A) × P(A) / P(B)")
    print("\nUse case: Update beliefs with new evidence")
    print("Example: Medical diagnosis, spam detection, etc.\n")
    
    # Medical test example
    print("Medical Test Example:")
    print("  • Disease prevalence: 1%")
    print("  • Test sensitivity (true positive): 95%")
    print("  • Test specificity (true negative): 90%")
    print("  • Question: If test is positive, what's P(Disease)?\n")
    
    p_disease = 0.01
    p_no_disease = 0.99
    p_pos_given_disease = 0.95  # Sensitivity
    p_pos_given_no_disease = 0.10  # 1 - Specificity
    
    # P(Positive)
    p_positive = (p_pos_given_disease * p_disease + 
                  p_pos_given_no_disease * p_no_disease)
    
    # Bayes: P(Disease | Positive)
    p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive
    
    print("Calculation:")
    print(f"  P(Positive) = {p_positive:.4f}")
    print(f"  P(Disease | Positive) = {p_disease_given_pos:.4f} = {p_disease_given_pos*100:.1f}%")
    print("\nSurprising result! Even with positive test, only ~9% chance")
    print("because disease is rare (base rate matters!)")


def demo_distributions():
    """Demonstrate probability distributions"""
    section_divider("3. PROBABILITY DISTRIBUTIONS")
    
    print("Distribution: Function describing probability of outcomes\n")
    
    # Uniform distribution
    print("--- Uniform Distribution ---")
    print("All outcomes equally likely")
    np.random.seed(42)
    uniform = np.random.uniform(0, 1, 10000)
    
    print(f"Generated {len(uniform)} samples from U(0,1)")
    print(f"Mean: {np.mean(uniform):.3f} (expected: 0.5)")
    print(f"Min: {np.min(uniform):.3f}")
    print(f"Max: {np.max(uniform):.3f}")
    
    # Normal (Gaussian) distribution
    print("\n--- Normal Distribution (Gaussian) ---")
    print("Bell curve: N(μ, σ²)")
    print("  • μ (mu) = mean")
    print("  • σ (sigma) = standard deviation")
    print("  • σ² = variance\n")
    
    mu = 0
    sigma = 1
    normal = np.random.normal(mu, sigma, 10000)
    
    print(f"Generated {len(normal)} samples from N(0, 1)")
    print(f"Sample mean: {np.mean(normal):.3f} (expected: 0)")
    print(f"Sample std: {np.std(normal):.3f} (expected: 1)")
    
    # 68-95-99.7 rule
    within_1_std = np.sum((normal >= -1) & (normal <= 1)) / len(normal)
    within_2_std = np.sum((normal >= -2) & (normal <= 2)) / len(normal)
    within_3_std = np.sum((normal >= -3) & (normal <= 3)) / len(normal)
    
    print(f"\n68-95-99.7 Rule (Empirical Rule):")
    print(f"  Within 1σ: {within_1_std*100:.1f}% (expected: ~68%)")
    print(f"  Within 2σ: {within_2_std*100:.1f}% (expected: ~95%)")
    print(f"  Within 3σ: {within_3_std*100:.1f}% (expected: ~99.7%)")
    
    # Binomial distribution
    print("\n--- Binomial Distribution ---")
    print("Number of successes in n trials")
    print("Example: Number of heads in 10 coin flips\n")
    
    n_trials = 10
    p_success = 0.5
    n_experiments = 10000
    
    binomial = np.random.binomial(n_trials, p_success, n_experiments)
    
    print(f"Experiment: {n_trials} coin flips, repeated {n_experiments} times")
    print(f"Mean heads per experiment: {np.mean(binomial):.2f}")
    print(f"Expected: n × p = {n_trials * p_success}")
    print(f"\nDistribution of results:")
    for i in range(n_trials + 1):
        count = np.sum(binomial == i)
        prob = count / n_experiments
        bar = "█" * int(prob * 100)
        print(f"  {i:2d} heads: {prob:.3f} {bar}")


def demo_descriptive_statistics():
    """Demonstrate descriptive statistics"""
    section_divider("4. DESCRIPTIVE STATISTICS")
    
    print("Descriptive Statistics: Summarize and describe data\n")
    
    # Sample data: Student test scores
    np.random.seed(42)
    scores = np.array([78, 82, 85, 88, 90, 92, 75, 95, 88, 91, 
                       84, 87, 89, 93, 86, 81, 94, 88, 85, 90])
    
    print("Dataset: Student test scores")
    print(f"Data: {scores[:10]}... (showing first 10)")
    print(f"Total samples: {len(scores)}\n")
    
    # Measures of central tendency
    print("--- Measures of Central Tendency ---")
    mean = np.mean(scores)
    median = np.median(scores)
    mode_data = Counter(scores)
    mode = mode_data.most_common(1)[0][0]
    
    print(f"Mean (average): {mean:.2f}")
    print(f"  Sum of values / count")
    print(f"  Used when: Data is symmetric, no outliers")
    
    print(f"\nMedian (middle value): {median:.2f}")
    print(f"  Middle value when sorted")
    print(f"  Used when: Data has outliers")
    
    print(f"\nMode (most frequent): {mode}")
    print(f"  Most common value")
    print(f"  Used when: Finding typical category")
    
    # Measures of spread
    print("\n--- Measures of Spread (Variability) ---")
    range_val = np.max(scores) - np.min(scores)
    variance = np.var(scores)
    std_dev = np.std(scores)
    
    print(f"Range: {range_val}")
    print(f"  Max - Min = {np.max(scores)} - {np.min(scores)}")
    print(f"  Simple but affected by outliers")
    
    print(f"\nVariance (σ²): {variance:.2f}")
    print(f"  Average squared deviation from mean")
    print(f"  Formula: Σ(x - μ)² / n")
    
    print(f"\nStandard Deviation (σ): {std_dev:.2f}")
    print(f"  Square root of variance")
    print(f"  Same units as original data")
    print(f"  Most commonly used measure of spread")
    
    # Quartiles and IQR
    print("\n--- Quartiles and Percentiles ---")
    q1 = np.percentile(scores, 25)
    q2 = np.percentile(scores, 50)  # Same as median
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    
    print(f"Q1 (25th percentile): {q1:.1f}")
    print(f"Q2 (50th percentile/Median): {q2:.1f}")
    print(f"Q3 (75th percentile): {q3:.1f}")
    print(f"IQR (Interquartile Range): {iqr:.1f}")
    print(f"  Q3 - Q1 = middle 50% of data")
    
    # Five-number summary
    print("\n--- Five-Number Summary ---")
    print(f"Minimum: {np.min(scores)}")
    print(f"Q1: {q1:.1f}")
    print(f"Median: {median:.1f}")
    print(f"Q3: {q3:.1f}")
    print(f"Maximum: {np.max(scores)}")


def demo_data_relationships():
    """Demonstrate relationships between variables"""
    section_divider("5. RELATIONSHIPS BETWEEN VARIABLES")
    
    print("Measuring how two variables relate to each other\n")
    
    # Generate correlated data
    np.random.seed(42)
    study_hours = np.array([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 
                           6, 7, 7, 8, 8, 9, 9, 10, 10, 11])
    # Test scores increase with study hours (with noise)
    test_scores = 50 + 3.5 * study_hours + np.random.normal(0, 3, len(study_hours))
    
    print("Dataset: Study hours vs Test scores")
    print(f"Study hours: {study_hours[:10]}...")
    print(f"Test scores: {test_scores[:10].astype(int)}...\n")
    
    # Covariance
    print("--- Covariance ---")
    cov = np.cov(study_hours, test_scores)[0, 1]
    print(f"Covariance: {cov:.2f}")
    print("  Measures how two variables vary together")
    print("  Positive: Both increase together")
    print("  Negative: One increases, other decreases")
    print("  Problem: Scale-dependent (hard to interpret)")
    
    # Correlation
    print("\n--- Correlation (Pearson's r) ---")
    corr = np.corrcoef(study_hours, test_scores)[0, 1]
    print(f"Correlation coefficient: {corr:.3f}")
    print("  Normalized covariance: ranges from -1 to +1")
    print("  +1: Perfect positive correlation")
    print("  0: No correlation")
    print("  -1: Perfect negative correlation")
    print(f"\nInterpretation:")
    if abs(corr) > 0.8:
        print(f"  Strong {'positive' if corr > 0 else 'negative'} correlation")
    elif abs(corr) > 0.5:
        print(f"  Moderate {'positive' if corr > 0 else 'negative'} correlation")
    else:
        print(f"  Weak correlation")
    
    print(f"\n  r² = {corr**2:.3f}")
    print(f"  {corr**2*100:.1f}% of variance in scores explained by study hours")
    
    # Important note
    print("\n⚠️  IMPORTANT: Correlation ≠ Causation")
    print("  High correlation doesn't mean one causes the other!")
    print("  Example: Ice cream sales & drownings correlate")
    print("  But ice cream doesn't cause drowning (both increase in summer)")


def demo_sampling():
    """Demonstrate sampling concepts"""
    section_divider("6. SAMPLING & ESTIMATION")
    
    print("Sampling: Using a subset to understand the whole population\n")
    
    # Create a population
    np.random.seed(42)
    population_size = 100000
    population = np.random.normal(100, 15, population_size)  # IQ scores
    
    true_mean = np.mean(population)
    true_std = np.std(population)
    
    print(f"Population: {population_size:,} people")
    print(f"True mean: {true_mean:.2f}")
    print(f"True std: {true_std:.2f}\n")
    
    # Take samples
    print("--- Taking Random Samples ---")
    sample_sizes = [10, 30, 100, 1000]
    
    for n in sample_sizes:
        sample = np.random.choice(population, n, replace=False)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample)
        error = abs(sample_mean - true_mean)
        
        print(f"\nSample size n={n:4d}:")
        print(f"  Sample mean: {sample_mean:.2f}")
        print(f"  Error: {error:.2f}")
        print(f"  Sample std: {sample_std:.2f}")
    
    print("\nObservation: Larger samples → better estimates!")
    
    # Central Limit Theorem
    print("\n--- Central Limit Theorem (CLT) ---")
    print("Key insight: Sample means are normally distributed")
    print("  Even if population is not normal!")
    print("  Larger sample → narrower distribution\n")
    
    # Demonstrate CLT
    n_samples = 1000
    sample_size = 30
    sample_means = []
    
    for _ in range(n_samples):
        sample = np.random.choice(population, sample_size, replace=False)
        sample_means.append(np.mean(sample))
    
    sample_means = np.array(sample_means)
    
    print(f"Took {n_samples} samples of size {sample_size}")
    print(f"Mean of sample means: {np.mean(sample_means):.2f}")
    print(f"True population mean: {true_mean:.2f}")
    print(f"Std of sample means: {np.std(sample_means):.2f}")
    print(f"Expected (SEM): {true_std / np.sqrt(sample_size):.2f}")
    print("  SEM = σ / √n (Standard Error of Mean)")
    
    # Confidence intervals
    print("\n--- Confidence Intervals ---")
    sample = np.random.choice(population, 100, replace=False)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    n = len(sample)
    
    # 95% confidence interval (using z-score ≈ 1.96)
    margin_of_error = 1.96 * (sample_std / np.sqrt(n))
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    print(f"Sample (n={n}): mean = {sample_mean:.2f}")
    print(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"True mean {true_mean:.2f} is {'inside' if ci_lower <= true_mean <= ci_upper else 'outside'} CI")
    print("\nInterpretation: 95% confident true mean is in this range")


def demo_hypothesis_testing():
    """Demonstrate hypothesis testing concepts"""
    section_divider("7. HYPOTHESIS TESTING")
    
    print("Hypothesis Testing: Make decisions using data\n")
    
    print("--- Basic Framework ---")
    print("1. Null Hypothesis (H₀): Default assumption (no effect)")
    print("2. Alternative Hypothesis (H₁): What we want to prove")
    print("3. Significance Level (α): Usually 0.05 (5%)")
    print("4. p-value: Probability of seeing data if H₀ is true")
    print("5. Decision: If p-value < α, reject H₀\n")
    
    # Example: A/B testing
    print("--- Example: A/B Testing (Two Websites) ---")
    print("\nScenario: Testing two website designs")
    print("  H₀: Both designs have same conversion rate")
    print("  H₁: Design B has higher conversion rate\n")
    
    np.random.seed(42)
    
    # Simulate data
    n_a = 1000
    n_b = 1000
    
    # Design A: 10% conversion
    conversions_a = np.random.binomial(1, 0.10, n_a)
    
    # Design B: 12% conversion (actually better)
    conversions_b = np.random.binomial(1, 0.12, n_b)
    
    rate_a = np.mean(conversions_a)
    rate_b = np.mean(conversions_b)
    
    print(f"Design A: {n_a} visitors, {np.sum(conversions_a)} conversions ({rate_a:.1%})")
    print(f"Design B: {n_b} visitors, {np.sum(conversions_b)} conversions ({rate_b:.1%})")
    print(f"Difference: {(rate_b - rate_a):.1%}")
    
    # Simple z-test (simplified)
    p_pooled = (np.sum(conversions_a) + np.sum(conversions_b)) / (n_a + n_b)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
    z_score = (rate_b - rate_a) / se
    
    # Approximate p-value (one-tailed)
    from scipy import stats
    try:
        p_value = 1 - stats.norm.cdf(z_score)
        print(f"\nz-score: {z_score:.3f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("✓ p-value < 0.05: Reject H₀")
            print("  Evidence that Design B is better!")
        else:
            print("✗ p-value ≥ 0.05: Fail to reject H₀")
            print("  Not enough evidence of difference")
    except:
        print("\n(scipy not available, skipping p-value calculation)")
        print(f"z-score: {z_score:.3f}")
        if z_score > 1.96:
            print("✓ z > 1.96: Statistically significant at α=0.05")
        else:
            print("✗ z ≤ 1.96: Not statistically significant")
    
    # Type I and Type II errors
    print("\n--- Errors in Hypothesis Testing ---")
    print("Type I Error (False Positive):")
    print("  Reject H₀ when it's actually true")
    print("  Example: Say design B is better when it's not")
    print(f"  Probability: α = 0.05 (5%)")
    
    print("\nType II Error (False Negative):")
    print("  Fail to reject H₀ when H₁ is true")
    print("  Example: Say no difference when B is actually better")
    print("  Probability: β (depends on sample size, effect size)")
    
    print("\nPower = 1 - β")
    print("  Probability of correctly detecting a real effect")


def demo_ml_applications():
    """Demonstrate ML applications of probability & statistics"""
    section_divider("8. MACHINE LEARNING APPLICATIONS")
    
    print("How Probability & Statistics are used in ML:\n")
    
    # 1. Data preprocessing
    print("--- 1. Data Preprocessing & Normalization ---")
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)
    
    print("Original data:")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Std: {np.std(data):.2f}")
    print(f"  Range: [{np.min(data):.2f}, {np.max(data):.2f}]")
    
    # Z-score normalization
    normalized = (data - np.mean(data)) / np.std(data)
    
    print("\nAfter Z-score normalization:")
    print(f"  Mean: {np.mean(normalized):.2f}")
    print(f"  Std: {np.std(normalized):.2f}")
    print(f"  Range: [{np.min(normalized):.2f}, {np.max(normalized):.2f}]")
    print("  → Data now has mean=0, std=1 (helps ML algorithms)")
    
    # 2. Train/test split
    print("\n--- 2. Train/Test Split ---")
    total_samples = 1000
    train_ratio = 0.8
    
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    split_point = int(total_samples * train_ratio)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    print(f"Total samples: {total_samples}")
    print(f"Training set: {len(train_indices)} samples ({train_ratio:.0%})")
    print(f"Test set: {len(test_indices)} samples ({1-train_ratio:.0%})")
    print("  → Randomly sample to avoid bias")
    
    # 3. Evaluation metrics
    print("\n--- 3. Model Evaluation (Classification) ---")
    print("Confusion Matrix:")
    print("                 Predicted")
    print("               Pos    Neg")
    print("  Actual Pos    TP     FN")
    print("         Neg    FP     TN\n")
    
    # Simulated predictions
    tp, fp, tn, fn = 85, 10, 90, 15
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}\n")
    
    print(f"Accuracy: {accuracy:.3f} = (TP + TN) / Total")
    print(f"Precision: {precision:.3f} = TP / (TP + FP)")
    print(f"  Of all positive predictions, how many correct?")
    print(f"Recall: {recall:.3f} = TP / (TP + FN)")
    print(f"  Of all actual positives, how many found?")
    print(f"F1-Score: {f1:.3f} = Harmonic mean of precision & recall")
    
    # 4. Naive Bayes classifier
    print("\n--- 4. Naive Bayes Classifier ---")
    print("Uses Bayes' theorem for classification")
    print("P(Class | Features) ∝ P(Features | Class) × P(Class)")
    print("\nExample: Spam detection")
    print("  Given words in email, calculate P(Spam | Words)")
    print("  Choose class with highest probability")
    
    # 5. Gaussian Naive Bayes
    print("\n--- 5. Feature Distributions ---")
    print("Assume features follow normal distribution")
    print("Example: Height distribution for gender classification")
    print("  Male: N(μ=175cm, σ=7cm)")
    print("  Female: N(μ=162cm, σ=6cm)")
    print("\nGiven height=170cm, which gender more likely?")
    
    from scipy import stats
    try:
        height = 170
        p_male = stats.norm.pdf(height, 175, 7)
        p_female = stats.norm.pdf(height, 162, 6)
        
        print(f"  P(Height=170 | Male) ∝ {p_male:.6f}")
        print(f"  P(Height=170 | Female) ∝ {p_female:.6f}")
        
        if p_male > p_female:
            print("  → More likely Male")
        else:
            print("  → More likely Female")
    except:
        print("  (scipy not available for calculation)")


def demo_practice_exercises():
    """Provide practice exercises"""
    section_divider("9. PRACTICE EXERCISES")
    
    print("Exercise 1: Calculate Basic Statistics")
    print("-" * 70)
    data = np.array([23, 25, 28, 30, 32, 35, 38, 40, 42, 45])
    print(f"Data: {data}")
    print("\nCalculate:")
    print("  1. Mean")
    print("  2. Median")
    print("  3. Standard deviation")
    print("  4. Q1, Q3, and IQR")
    
    print("\nSolution:")
    print(f"  Mean: {np.mean(data):.2f}")
    print(f"  Median: {np.median(data):.2f}")
    print(f"  Std Dev: {np.std(data):.2f}")
    print(f"  Q1: {np.percentile(data, 25):.2f}")
    print(f"  Q3: {np.percentile(data, 75):.2f}")
    print(f"  IQR: {np.percentile(data, 75) - np.percentile(data, 25):.2f}")
    
    print("\n" + "=" * 70)
    print("\nExercise 2: Probability Calculation")
    print("-" * 70)
    print("Rolling two dice:")
    print("  What's the probability of getting sum = 7?")
    
    print("\nSolution:")
    outcomes = [(i, j) for i in range(1, 7) for j in range(1, 7)]
    sum_7 = [(i, j) for i, j in outcomes if i + j == 7]
    prob = len(sum_7) / len(outcomes)
    
    print(f"  Total outcomes: {len(outcomes)}")
    print(f"  Favorable outcomes (sum=7): {len(sum_7)}")
    print(f"  Outcomes: {sum_7}")
    print(f"  P(Sum=7) = {prob:.4f} = {prob:.2%}")
    
    print("\n" + "=" * 70)
    print("\nExercise 3: Correlation")
    print("-" * 70)
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    
    print(f"X: {x}")
    print(f"Y: {y}")
    print("\nCalculate correlation coefficient")
    
    print("\nSolution:")
    corr = np.corrcoef(x, y)[0, 1]
    print(f"  r = {corr:.4f}")
    if corr > 0.7:
        print("  Strong positive correlation")
    elif corr > 0.3:
        print("  Moderate positive correlation")
    else:
        print("  Weak correlation")


def demo_key_takeaways():
    """Summarize key concepts"""
    section_divider("KEY TAKEAWAYS")
    
    print("Essential Concepts for Machine Learning:\n")
    
    print("1. PROBABILITY")
    print("   • Basic rules: sum, product, complement")
    print("   • Conditional probability: P(A|B)")
    print("   • Bayes' theorem: Update beliefs with evidence\n")
    
    print("2. DISTRIBUTIONS")
    print("   • Uniform: All outcomes equally likely")
    print("   • Normal (Gaussian): Bell curve, most common in nature")
    print("   • Binomial: Success/failure in n trials\n")
    
    print("3. DESCRIPTIVE STATISTICS")
    print("   • Central tendency: mean, median, mode")
    print("   • Spread: variance, standard deviation, IQR")
    print("   • Percentiles and quartiles\n")
    
    print("4. RELATIONSHIPS")
    print("   • Covariance: How variables vary together")
    print("   • Correlation: Normalized measure (-1 to +1)")
    print("   • Remember: Correlation ≠ Causation!\n")
    
    print("5. SAMPLING & INFERENCE")
    print("   • Sample statistics estimate population parameters")
    print("   • Central Limit Theorem: Sample means are normal")
    print("   • Confidence intervals: Range of likely values\n")
    
    print("6. HYPOTHESIS TESTING")
    print("   • Null vs Alternative hypothesis")
    print("   • p-value: Evidence against null")
    print("   • Type I & II errors\n")
    
    print("7. ML APPLICATIONS")
    print("   • Data normalization (z-score)")
    print("   • Train/test splitting")
    print("   • Evaluation metrics (accuracy, precision, recall)")
    print("   • Probabilistic classifiers (Naive Bayes)")
    print("   • Feature distributions")


def main():
    """Main function to run all demonstrations"""
    print("\n" + "★" * 70)
    print("  PROBABILITY & STATISTICS FOR MACHINE LEARNING")
    print("  Essential Foundations for Data Science")
    print("  Python Examples with NumPy")
    print("★" * 70)
    
    # Run all demonstrations
    demo_basic_probability()
    demo_conditional_probability()
    demo_distributions()
    demo_descriptive_statistics()
    demo_data_relationships()
    demo_sampling()
    demo_hypothesis_testing()
    demo_ml_applications()
    demo_practice_exercises()
    demo_key_takeaways()
    
    # Final message
    section_divider("COMPLETE!")
    print("You've completed Probability & Statistics basics!")
    print("\nWhat you've learned:")
    print("  ✓ Probability fundamentals and conditional probability")
    print("  ✓ Key distributions (uniform, normal, binomial)")
    print("  ✓ Descriptive statistics (mean, std, correlation)")
    print("  ✓ Sampling, estimation, and confidence intervals")
    print("  ✓ Hypothesis testing framework")
    print("  ✓ How statistics powers ML algorithms")
    print("\nNext steps:")
    print("  1. Practice with real datasets")
    print("  2. Move to Core ML algorithms")
    print("  3. Start with Linear Regression")
    print("  4. Apply these concepts to real problems")
    print("\nYou're now ready for machine learning algorithms!")
    print("\n" + "★" * 70 + "\n")


if __name__ == "__main__":
    main()
