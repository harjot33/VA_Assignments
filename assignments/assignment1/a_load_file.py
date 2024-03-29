from pathlib import Path
import pandas as pd


##############################################
# Implement the below method
# The method should be dataset-independent
##############################################
def read_dataset(path: Path) -> pd.DataFrame:
    """
    This method will be responsible to read the dataset.
    Please implement this method so that it returns a pandas dataframe from a given path.
    Notice that this path is of type Path, which is a helper type from python to best handle
    the paths styles of different operating systems.
    """

    df = pd.read_csv(path)
    print(df)
    return df

if __name__ == "__main__":
    """
    In case you don't know, this if statement lets us only execute the following lines
    if and only if this file is the one being executed as the main script. Use this
    in the future to test your scripts before integrating them with other scripts.
    """

    # Loading the dataset
    dataset = read_dataset(Path(r'H:\New folder (2)\Assignments (3)\Assignments', 'iris.csv'))
    assert type(dataset) == pd.DataFrame
