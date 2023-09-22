import numpy as np
from scipy.stats import truncnorm

def gibbs_sampling(N_iterations , s1_s2_mean_col, s_covar_matrix, y):
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
    A = np.array([[1, -1]]) # Matrix A
    s1 = float(s1_s2_mean_col[0])
    s2 = float(s1_s2_mean_col[1])
    #print(f's1: {s1}, s2: {s2}')

    if y == 1:
        a = 0
        b = np.inf
    elif y == -1:
        a = -np.inf
        b = 0
    else:
        raise ValueError("Error: y must be 1 or -1")

    s1_mean_list, s2_mean_list, s_covar_matrix1, s_covar_matrix2, s1_list, s2_list = [], [], [], [], [], []

    for i in range(N_iterations):
        # Sample t from p(t|s1,s2,y)
        mean_t = s1 - s2
        conditional_Vt = s_covar_matrix[0,0] + s_covar_matrix[1,1]
        t = truncnorm.rvs((a - mean_t) / np.sqrt(conditional_Vt), (b - mean_t) / np.sqrt(conditional_Vt), loc=mean_t, scale=np.sqrt(conditional_Vt))

        # Sample s1 and s2 from multivariate normal, p(s1,s2|t>0)
        s_covar_matrix_old = s_covar_matrix
        s_covar_matrix = np.linalg.inv(np.linalg.inv(s_covar_matrix) + np.transpose(A) @A*(1/5))
        s1_s2_mean_col = s_covar_matrix @ (np.linalg.inv(s_covar_matrix_old) @ s1_s2_mean_col + np.transpose(A) * (1/5) * t)

        s1, s2 = np.random.multivariate_normal(s1_s2_mean_col.flatten(), s_covar_matrix, 1).T

        # Save values to lists
        s1_mean_list.append(float(s1_s2_mean_col[0]))
        s2_mean_list.append(float(s1_s2_mean_col[1]))
        s_covar_matrix1.append(float(s_covar_matrix[0][0]))
        s_covar_matrix2.append(float(s_covar_matrix[1][1]))
        s1_list.append(float(s1))
        s2_list.append(float(s2))

    return s1_mean_list, s2_mean_list, s_covar_matrix1, s_covar_matrix2, s1_list, s2_list







# Optional: To test the function

if __name__ == "__main__":
    # Make sure you define the required parameters: samples, s1_s2_mean_col, s_covar_matrix, A, a, b, Vt
    samples = 3000
    s1_s2_mean_col = np.array([[25], [25]])
    s_covar_matrix = np.array([[8, 0], [0, 8]])

    a = 0
    b = np.inf
    y = 1
    

    s1_means, s2_means, s1_vars, s2_vars, s1_samples, s2_samples = gibbs_sampling(samples, s1_s2_mean_col, s_covar_matrix, y)
    print(len(s1_samples))
    print(len(s2_samples))
    #print(s1_samples)

    # Plot the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.plot(range(samples), s1_samples)
    plt.title("s1 samples")
    plt.subplot(2, 2, 2)
    plt.plot(s2_samples)
    plt.title("s2 samples")
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.hist(s1_samples, bins=50)
    plt.title("s1 histogram")
    plt.show()

