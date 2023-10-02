# APML - Advanced probabilistic machine learning Project
# Preprecessing
# prepare the dataset
import pandas as pd

def preprocess_serieA_no_draws(games_df, dataset_name):
    """
    Load and preprocess the Serie A dataset.

    Parameters:
    - filepath (str): Path to the dataset.

    Returns:
    - games_df (pd.DataFrame): Processed dataframe without draw games.
    """
    if dataset_name == 'SerieA':

        # Remove draw games
        draw_games_df = games_df[games_df['score1'] == games_df['score2']]
        games_df = games_df.drop(draw_games_df.index)

        # Create a column with the score difference of the game
        games_df['score_diff'] = games_df.apply(lambda row: row['score1'] - row['score2'], axis=1)

        # Remove unnecessary columns (score1, score2, HH:MM, yyyy-mm-dd)
        games_df = games_df.drop(['score1', 'score2', 'HH:MM', 'yyyy-mm-dd'], axis=1)

    elif dataset_name == 'csgo':
        
        # Remove draw games
        draw_games_df = games_df[games_df['result_1'] == games_df['result_2']]
        games_df = games_df.drop(draw_games_df.index)

        # Create a column with the score difference of the game
        games_df['score_diff'] = games_df.apply(lambda row: row['result_1'] - row['result_2'], axis=1)

        # Remove unnecessary columns (score1, score2, HH:MM, yyyy-mm-dd)
        games_df = games_df[['team_1', 'team_2', 'score_diff', 'map_winner']]
        


    return games_df





# Optional: If you want to run the function and see its result directly from this script
if __name__ == "__main__":
    serieA_data = pd.read_csv('SerieA_dataset.csv', header=0)
    df = preprocess_serieA_no_draws(serieA_data)

