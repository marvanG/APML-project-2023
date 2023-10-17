# Prediction function (deterministic)
# if skills are equal, team 1 wins, might need to change this
def prediction(team1_mean, team2_mean, game_type):

    if game_type == 'football':
        home_team_advantage = 1.05

        if team1_mean*home_team_advantage >= team2_mean:
            return 1
        elif team1_mean*home_team_advantage < team2_mean:
            return -1
        else:
            print('Error in prediction function')
            return ValueError
    
    elif game_type == 'csgo':
        if team1_mean >= team2_mean:
            return 1
        elif team1_mean < team2_mean:
            return -1
        else:
            print('Error in prediction function')

            return ValueError
    elif game_type == 'basketball':
        print('Basketball is Not implemented yet')
        return ValueError
    else:
        print('Error in prediction function, game_type not recognized')
        return ValueError
