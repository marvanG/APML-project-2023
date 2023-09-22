# APML - Advanced probabilistic machine learning Project
# Preprecessing
# prepare the dataset
import pandas as pd

def preprocess_serieA_no_draws(games_df):
    """
    Load and preprocess the Serie A dataset.

    Parameters:
    - filepath (str): Path to the dataset.

    Returns:
    - games_df (pd.DataFrame): Processed dataframe without draw games.
    """

    # print(f'Shape of the games dataset: {games_df.shape}')
    # print(f'first 3 rows:\n{games_df.head(3)}')
    # print('\nDescription of the games dataset:') 
    # print(games_df.describe())

    # Remove draw games
    draw_games_df = games_df[games_df['score1'] == games_df['score2']]
    games_df = games_df.drop(draw_games_df.index)
    # print(f'\nShape of the draw games dataset: {draw_games_df.shape}')
    # print(f'first 3 rows of draw games:\n{draw_games_df.head(3)}')
    # print(f'\nShape of the games dataset after removing draw games: {games_df.shape}')

    # Create a new column with the result of the game
    games_df['y'] = games_df.apply(lambda row: 1 if row['score1'] > row['score2'] else -1, axis=1)
    # print(f'first 3 rows of no-draw games:\n{games_df.head(3)}')

    # Remove unnecessary columns (score1, score2, HH:MM, yyyy-mm-dd)
    games_df = games_df.drop(['score1', 'score2', 'HH:MM', 'yyyy-mm-dd'], axis=1)
    # print(f'\nShape of the games dataset after removing unnecessary columns: {games_df.shape}')
    # print(f'first 3 rows of no-draw games:\n{games_df.head(3)}')

    return games_df





# Optional: If you want to run the function and see its result directly from this script
if __name__ == "__main__":
    serieA_data = pd.read_csv('SerieA_dataset.csv', header=0)
    df = preprocess_serieA_no_draws(serieA_data)

