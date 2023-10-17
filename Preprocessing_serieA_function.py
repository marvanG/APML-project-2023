# APML - Advanced probabilistic machine learning Project
# Preprecessing
# prepare the dataset
import pandas as pd
from numpy import sign

def preprocess_serieA_no_draws(df, dataset_name, remove_draws):
    """
    Load and preprocess the Serie A dataset.

    Parameters:
    - filepath (str): Path to the dataset.

    Returns:
    - df (pd.DataFrame): Processed dataframe without draw games.
    """
    if dataset_name == 'SerieA':
        if remove_draws:

             # Remove draw games
             draw_games_df = df[df['score1'] == df['score2']]
             df = df.drop(draw_games_df.index)

        # Create a column with the score difference of the game
        df['score_diff'] = df.apply(lambda row: row['score1'] - row['score2'], axis=1)

        # create a column with the winner of the game, 0 if draw, 1 if team1 won, -1 if team2 won
        df['y'] = df['score_diff'].apply(lambda x: 0 if x == 0 else sign(x))

        # Remove unnecessary columns (score1, score2, HH:MM, yyyy-mm-dd)
        df = df.drop(['score1', 'score2', 'HH:MM', 'yyyy-mm-dd'], axis=1)
        print(df.head())

    elif dataset_name == 'csgo':
        if remove_draws:
            
            # Remove draw games
            draw_games_df = df[df['result_1'] == df['result_2']]
            df = df.drop(draw_games_df.index)

            # Create a column with the score difference of the game
            df['score_diff'] = df.apply(lambda row: row['result_1'] - row['result_2'], axis=1)

            # Remove unnecessary columns (score1, score2, HH:MM, yyyy-mm-dd)
            df = df[['team_1', 'team_2', 'score_diff', 'map_winner']]

            # Rename columns
            df = df.rename(columns={'team_1': 'team1', 'team_2': 'team2', 'map_winner': 'y'})

            df['y'] = df['y'].apply(lambda x: 1 if x == 1 else -1)
            print(df.head())

        else:
            print('Error in dataset name, dataset not recognized')
            return ValueError
    
    return df





# Optional: If you want to run the function and see its result directly from this script
if __name__ == "__main__":
    serieA_data = pd.read_csv('SerieA_dataset.csv', header=0)
    
    df = preprocess_serieA_no_draws(serieA_data, dataset_name='SerieA', remove_draws=False)

