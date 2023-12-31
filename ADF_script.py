# Assumed density filtering (ADF)
from prediction_script import prediction
from Gibbs_sampler_function import gibbs_sampling
import numpy as np
def ADF(teams_dictionary, dataframe,extension):

    # Assumed density filtering (ADF)
    n_iter = 800
    t_var = 6
    burn_in = 250

    # One step ahead prediction
    predictions = []
    # Loop over all games

    for index, game1 in dataframe.iterrows():
        
        # Load means and variances from dictionary
        mean_team1 = float(teams_dictionary[game1[0]][0])
        mean_team2 = float(teams_dictionary[game1[1]][0])

        variance_team1 = float(teams_dictionary[game1[0]][1])
        variance_team2 = float(teams_dictionary[game1[1]][1])

        # One-Step-Ahead prediction
        y = np.sign(game1['score_diff'])
        if abs(y) == 1:
            y_pred = prediction(mean_team1, mean_team2, game_type='football')
            predictions.append(y_pred)
        elif y == 0:
            y_pred = None
        
        # create mean column and covariance matrix
        s1_s2_mean_col = np.array([[mean_team1, mean_team2]]).reshape(-1,1)
        s_cov_matrix = np.array([[variance_team1, 0], [0, variance_team2]])

        score_difference = game1['score_diff']
        

        s1_samples, s2_samples = gibbs_sampling(n_iter, s1_s2_mean_col, s_cov_matrix, t_var, score_difference,extension=extension)

        # results
        s1_samples = s1_samples[burn_in:]
        s2_samples = s2_samples[burn_in:]

        s1_mean = np.mean(s1_samples)
        s2_mean = np.mean(s2_samples)
        s1_var = np.var(s1_samples)
        s2_var = np.var(s2_samples)

        # Update team dictionary means
        teams_dictionary[game1[0]][0] = s1_mean
        teams_dictionary[game1[1]][0] = s2_mean

        # update team dictionary variances
        teams_dictionary[game1[0]][1] = s1_var
        teams_dictionary[game1[1]][1] = s2_var

        # Print progress every 100 games
        if index % 30 == 0:
            print(f'\nCurrent game: {index}: team1 {game1[0]} with prior skill {mean_team1:.1f} and variance {variance_team1:.2f} vs team2 {game1[1]} with skill {mean_team2:.1f} and variance {variance_team2:.2f}')
            print(f'Prediction: {y_pred}, result : {np.sign(score_difference)}, score difference: {score_difference}')
            print(f'Updated skill team1: {s1_mean:.1f} and variance {s1_var:.2f} vs team2: {s2_mean:.1f} and variance {s2_var:.2f}')

    return teams_dictionary, predictions