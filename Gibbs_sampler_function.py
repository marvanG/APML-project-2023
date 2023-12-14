import numpy as np
from scipy.stats import truncnorm
from numpy import sign

def gibbs_sampling(N_iterations , s1_s2_mean_col, s_covar_matrix, t_Var, score_diff,extension):
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
    if score_diff == 0:
        y = 0
    else:
        y= sign(score_diff)
    
    A = np.array([[1, -1]]) # Matrix A
    s1 = float(s1_s2_mean_col[0])
    s2 = float(s1_s2_mean_col[1])
    s1_var = s_covar_matrix[0][0]
    s2_var = s_covar_matrix[1][1]
    s_sampling_covarmatrix = s_covar_matrix

    # Home team advantage parameter
    home_advantage_param = -0.4

    # score diff skill boost parameter
    if abs(score_diff) <= 1:
        score_diff_param = 0
    elif abs(score_diff) > 1:
        score_diff_param = score_diff
  
    else:
        print('Error in score_diff_param')
        return ValueError


    # Uncertainty offset parameter
    if (s1_var + s2_var) > 20:
        uncertainty_offset_param = 10
    elif (s1_var + s2_var) > 0.5 and (s1_var + s2_var) <= 20:
        uncertainty_offset_param = 5
    elif (s1_var + s2_var) <= 0.5:
        uncertainty_offset_param = 1.5
    
    
    # boost_parameter = home_advantage_param  / uncertainty_offset_param
    score_diff_param = score_diff_param / uncertainty_offset_param
    
    

    # Truncnorm parameters
    if y == 1: # Player 1 wins
        a = 0
        b = np.inf

    elif y == -1: # Player 2 wins
        a = -np.inf
        b = 0

    elif y == 0: # Draw
        a = -np.inf
        b = np.inf
    else:
        print("Error in y")
        return ValueError

    s1_list, s2_list = [], []
    

   
    t = 0

    s_old_var = s_sampling_covarmatrix
    s_sampling_covarmatrix = np.linalg.inv(np.linalg.inv(s_sampling_covarmatrix) + np.transpose(A) @A*(1/t_Var))
    s_mean_old = s1_s2_mean_col
    for i in range(N_iterations):

       

       
        s1_s2_mean_col = s_sampling_covarmatrix @ (np.linalg.inv(s_old_var) @ s_mean_old + np.transpose(A) * (1/t_Var) * t)
        s1, s2 = np.random.multivariate_normal(s1_s2_mean_col.flatten(), s_sampling_covarmatrix, 1).T 

        if extension:
            mean_t = (s1- s2) #+ score_diff_param + home_advantage_param # |--Q10 project extension = boost parameter--|
        else:
            mean_t = (s1- s2) 
            
        if y == 0:
            mean_t *=0.85

        t = truncnorm.rvs((a - mean_t) / np.sqrt(t_Var), (b - mean_t) / np.sqrt(t_Var), loc=mean_t, scale=np.sqrt(t_Var), size=1)
        # Sample s1 and s2 from multivariate normal, p(s1,s2|t,y)

        
 

        # Sample t from p(t|s1,s2,y)
        
        

        # Save values and lists
        s1_list.append(float(s1))
        s2_list.append(float(s2))
        
    return s1_list, s2_list



# Optional: To test the function

if __name__ == "__main__":
    # Make sure you define the required parameters: samples, s1_s2_mean_col, s_covar_matrix, A, a, b, Vt
    samples = 10000
    s1_s2_mean_col = np.array([[25], [25]])
    s_covar_matrix = np.array([[3, 0], [0, 3]])
    Vt = 30
    score_diff = 1

    s1_samples, s2_samples = gibbs_sampling(samples, s1_s2_mean_col, s_covar_matrix, Vt, score_diff, extension=False)

    burn_in = 0
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


    # Plot the results
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    print("\nPlotting the results...")
    plt.figure(1, figsize=(10, 6))

    # Plot s1
    plt.subplot(2, 2, 1)
    plt.plot(s1_samples, label="s1 samples")
    plt.xlabel("Iteration")
    plt.ylabel("s1")
    plt.legend()
    plt.title("s1 samples")

    # Plot s2
    plt.subplot(2, 2, 2)
    plt.plot(s2_samples, label="s2 samples")
    plt.xlabel("Iteration")
    plt.ylabel("s2")
    plt.legend()
    plt.title("s2 samples")

    plt.tight_layout()
    plt.show()

    # Historgram
    plt.figure(2, figsize=(10, 6))

    # Plot s1 histogram
    plt.subplot(2, 2, 1)
    plt.hist(s1_samples, bins=50, density=True)
    plt.title("s1 histogram")

    # Plot s1 pdf
    x = np.linspace(min(s1_samples)-5, max(s1_samples)+5, 100)
    plt.plot(x, norm.pdf(x, s1_mean, np.sqrt(s1_var)), label="s1 pdf", color="red")


    # Plot s2
    plt.subplot(2, 2, 2)
    plt.hist(s2_samples, bins=50, density=True)
    plt.title("s2 histogram")

    # Plot s2 pdf
    x = np.linspace(min(s2_samples)-5, max(s2_samples)+5, 100)
    plt.plot(x, norm.pdf(x, s2_mean, np.sqrt(s2_var)), label="s2 pdf", color="red")
    plt.tight_layout()
    plt.show()


