# Create team dictionary
def df_to_dictionary(df):
    mean_skill = 25
    variance_skill = 64
    unique_teams_1 = df['team1'].unique().tolist()
    unique_teams_2 = df['team2'].unique().tolist()
    if len(unique_teams_1) != len(unique_teams_2):
        print('Error in data, some team is not present in both columns')
        return ValueError
    
    unique_teams = list(set(unique_teams_1 + unique_teams_2))
    dictionary = {i:[mean_skill, variance_skill] for i in unique_teams}

    return dictionary
