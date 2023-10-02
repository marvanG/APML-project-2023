import numpy as np
from scipy.stats import truncnorm
from numpy import sign

def gibbs_sampling(N_iterations , s1_s2_mean_col, s_covar_matrix, t_Var, score_diff):
    """
    Perform Gibbs sampling.

    Parameters:
    - N_iterations (int): Number of Gibbs sampling iterations.
    - s1_s2_mean_col (array-like): Initial mean values.
    - s_covar_matrix (array-like): Initial Variance values.
    - A (array-like): Matrix A ([1, -1]).
    - y: outcome of the game (1 or -1)
    - t_var (float): Variance for t.

    Returns:
    - s1_list, s2_list (list): List of sampled values for s1 and s2.
    """
   

    # Parameters
    # y = sign(score_diff)
    y= sign(score_diff)
    A = np.array([[1, -1]]) # Matrix A
    s1 = float(s1_s2_mean_col[0])
    s2 = float(s1_s2_mean_col[1])
    s1_var = s_covar_matrix[0][0]
    s2_var = s_covar_matrix[1][1]
    # Home team advantage parameter
    home_advantage_param = 0.5

    # score diff skill boost parameter
    if abs(score_diff) == 1:
        score_diff_param = 0
    elif abs(score_diff) > 1 and abs(score_diff) <= 5:
      score_diff_param = score_diff
    elif abs(score_diff) > 5:
        score_diff_param = 5*sign(score_diff)

    else:
        raise ValueError("Error: score_diff error")
    
    # Uncertainty offset parameter
    if (s1_var + s2_var) > 75:
        uncertainty_offset_param = 10
    elif (s1_var + s2_var) > 30 and (s1_var + s2_var) <= 75:
        uncertainty_offset_param = 5
    elif (s1_var + s2_var) > 15 and (s1_var + s2_var) <= 30:
        uncertainty_offset_param = 2.5
    elif (s1_var + s2_var) > 5 and (s1_var + s2_var) <= 15:
        uncertainty_offset_param = 2
    elif (s1_var + s2_var) <= 5:
        uncertainty_offset_param = 1
    
    
    boost_parameter = (-home_advantage_param + score_diff_param)/uncertainty_offset_param

    if y == 1: # Player 1 wins
        # truncnorm parameters
        a = 0
        b = np.inf

    elif y == -1: # Player 2 wins
        # truncnorm parameters
        a = -np.inf
        b = 0

    else:
        raise ValueError("Error: y must be 1 or -1")

    s1_list, s2_list = [], []
    s_sampling_covarmatrix = np.linalg.inv(np.linalg.inv(s_covar_matrix) + np.transpose(A) @A*(1/t_Var))

    for i in range(N_iterations):
        
        # Sample t from p(t|s1,s2,y)
        mean_t = (s1 - s2) #+ boost_parameter # |--Q10 project extension = boost parameter--|

        # print(f'mean_t: {mean_t}')
        t = truncnorm.rvs((a - mean_t) / np.sqrt(t_Var), (b - mean_t) / np.sqrt(t_Var), loc=mean_t, scale=np.sqrt(t_Var), size=1)

        # Sample s1 and s2 from multivariate normal, p(s1,s2|t,y)
        s_sampling_mean = s_sampling_covarmatrix @ (np.linalg.inv(s_covar_matrix) @ s1_s2_mean_col + np.transpose(A) * (1/t_Var) * t)
        s1, s2 = np.random.multivariate_normal(s_sampling_mean.flatten(), s_sampling_covarmatrix, 1).T 



        # Save values and lists
        s1_list.append(float(s1))
        s2_list.append(float(s2))
        
    return s1_list, s2_list



# s1_var = np.var(s1_list)
#     s2_var = np.var(s2_list)

#     s1_mean = np.mean(s1_list)
#     s2_mean = np.mean(s2_list)



# Optional: To test the function

if __name__ == "__main__":
    # Make sure you define the required parameters: samples, s1_s2_mean_col, s_covar_matrix, A, a, b, Vt
    samples = 10000
    s1_s2_mean_col = np.array([[25], [25]])
    s_covar_matrix = np.array([[64, 0], [0, 64]])
    Vt = 5
    score_diff = 6

    s1_samples, s2_samples = gibbs_sampling(samples, s1_s2_mean_col, s_covar_matrix, Vt, score_diff)

    burn_in = 500
    s1_samples = s1_samples[burn_in:]
    s2_samples = s2_samples[burn_in:]

    print(len(s1_samples))
    print(len(s2_samples))

    # Print the results
    s1_mean = np.mean(s1_samples)
    s2_mean = np.mean(s2_samples)
    s1_var = np.var(s1_samples)
    s2_var = np.var(s2_samples)
    print(f"\nMean of s1: {s1_mean}")
    print(f"\nMean of s2: {s2_mean}")
    print(f'\nVariance of s1: {s1_var}')
    print(f'\nVariance of s2: {s2_var}')


    # # Plot the results
    # import matplotlib.pyplot as plt
    # from scipy.stats import norm
    # print("\nPlotting the results...")
    # plt.figure(1, figsize=(10, 6))

    # # Plot s1
    # plt.subplot(2, 2, 1)
    # plt.plot(s1_samples, label="s1 samples")
    # plt.xlabel("Iteration")
    # plt.ylabel("s1")
    # plt.legend()
    # plt.title("s1 samples")

    # # Plot s2
    # plt.subplot(2, 2, 2)
    # plt.plot(s2_samples, label="s2 samples")
    # plt.xlabel("Iteration")
    # plt.ylabel("s2")
    # plt.legend()
    # plt.title("s2 samples")

    # plt.tight_layout()
    # plt.show()

    # # Historgram
    # plt.figure(2, figsize=(10, 6))

    # # Plot s1 histogram
    # plt.subplot(2, 2, 1)
    # plt.hist(s1_samples, bins=50, density=True)
    # plt.title("s1 histogram")

    # # Plot s1 pdf
    # x = np.linspace(min(s1_samples)-5, max(s1_samples)+5, 100)
    # plt.plot(x, norm.pdf(x, s1_mean, np.sqrt(s1_var)), label="s1 pdf", color="red")


    # # Plot s2
    # plt.subplot(2, 2, 2)
    # plt.hist(s2_samples, bins=50, density=True)
    # plt.title("s2 histogram")

    # # Plot s2 pdf
    # x = np.linspace(min(s2_samples)-5, max(s2_samples)+5, 100)
    # plt.plot(x, norm.pdf(x, s2_mean, np.sqrt(s2_var)), label="s2 pdf", color="red")
    # plt.tight_layout()
    # plt.show()


