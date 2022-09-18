import config

from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
import pandas as pd

def undersampling(df, apriori=0.10, target='is_applied'):

    # Get the indices per target value
    idx_0 = df[df[target] == 0].index
    idx_1 = df[df[target] == 1].index

    # Get original number of records per target value
    nb_0 = len(df.loc[idx_0])
    nb_1 = len(df.loc[idx_1])

    # Calculate the undersampling rate and resulting number of records with target=0
    undersampling_rate = ((1-apriori)*nb_1)/(nb_0*apriori)
    undersampled_nb_0 = int(undersampling_rate*nb_0)
    print('Rate to undersample records with target=0: {}'.format(undersampling_rate))
    print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))

    # Randomly select records with target=0 to get at the desired a priori
    undersampled_idx = shuffle(idx_0, random_state=0, n_samples=undersampled_nb_0)

    # Construct list with remaining indices
    idx_list = list(undersampled_idx) + list(idx_1)

    # Return undersample data frame
    undersampled_data = df.loc[idx_list].reset_index(drop=True)

    return undersampled_data

# def oversampling()

if __name__ == "__main__":

    df = pd.read_parquet(config.DATA + "train_impute.parquet")
    undersampled_data = undersampling(df)
    undersampled_data.to_parquet(config.DATA + "train_impute_undersampled.parquet")

