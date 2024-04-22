import json
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

"""
Term definitions:
A *criterium* 
A *characteristic* is an attribute that a person can have, e.g. being female, being in the age group 16-25, or having a hugh school education.
"""


def get_overview(sample: pd.DataFrame, criteria: dict[str]) -> dict[pd.DataFrame]:
    """Get an overview of how a given lotting compares to the desired criteria

    Args:
        sample (pd.DataFrame): A DataFrame containg a propoes lotting of participants
        criteria (dict[str]): A dictionary mapping a criterium (e.g. age group, gender, education) to the desired number of participants with each characteristic in the criterium (e.g. 10 men and 10 women)

    Returns:
        dict[pd.DataFrame]: A dictionary mapping each criterium to a DataFrame with columns "what we have", "what we want", and "difference", showing how many participants in *sample* have a given characteristic in the criterium, how many participants it is desired to have with each characeristic (given by *criteria*), and the difference between these values for each characteristic respectively.
    """
    overview: dict = {}
    # Iterate over the different criteria (e.g. age group, gender, education etc.)
    for criterium in criteria:
        # Pandas Series with the number of people in *sample* with all the different characteristics in the criterium (e.g. man: 15, woman: 15).
        have = pd.Series(sample[criterium].value_counts(), name="what we have")
        # Pandas Series containg how many people we want with all the different characteristics in the criterium. This is read directly from the *criterium*
        # dictionary.
        want = pd.Series(criteria[criterium]["values"], name="what we want")
        # Concatenate the two Series. If there are any characteristics that no people in *sample* have, these will first become NaN in the "what we
        # have" column, which we replace with 0.
        df_con = pd.concat([have, want], axis=1).fillna(0)
        # Calcluate the difference between the two columns and insert the values into a new column.
        df_con["difference"] = df_con["what we have"] - df_con["what we want"]
        # Add the DataFrame with the three columns to the overview dictionary, with the relevant criterium as the key.
        overview[criterium] = df_con
    return overview


def excess_or_demand(
    overview: dict[pd.DataFrame],
    population: pd.DataFrame,
    strictness: float = np.inf,
    demand: bool = True,
) -> list[pd.DataFrame]:
    """A function to find the people in a population that have characteristics that are either in demand or in excess. Whether it finds the former or the latter is determined by the *demand* toggle.

    Args:
        overview (dict[pd.DataFrame]): A dictionary mapping criteria to a DataFrame of characteristics. See the output of /get_overview/
        population (pd.DataFrame): The population from which the poeple with the relevant charcteristics are to be selected. When looking for people in demand, the population should be the total set of volunteers minus the volunteers already lotted. When looking for people in excess the population should be the people in the lotting.
        strictness (float, optional): The numbre of characteristics in demand/excess a person need to have to be considered in demand/excess. Defaults to all.
        demand (bool, optional): Toggle for whether to look for people in demnd or in excess. Defaults to True.

    Returns:
        list[pd.DataFrame]: A list of DataFrames of people in demand/excess for different groups of criteria.
    """
    masks = {}
    # Iterate over the criteria and the overview for each of them
    for criterium, df in overview.items():
        # Find the indecies of the characteristics that are either in demand or in excess depending on the value of *demand*
        subsample = (
            df[df["difference"] < 0].index if demand else df[df["difference"] > 0].index
        )
        # If there are any relevant characteristics, create a mask of the population that is True for people who have any of the characteristics and False for
        # people who have none. The mask is saved in the dictionary /masks/ whith the criterium as the key. Since we require that all criteria demand the same
        # total number of people, any criterium that has at least one characterisitc in demand _must_ also have at least one characterisitc in excess.
        if len(subsample.values) > 0:
            masks[criterium] = [
                any(x)
                for x in zip(*(population[criterium] == val for val in subsample))
            ]
    # Change the strictness to the minimum of the original strictness and the number of masks. The number of masks is equal to the number of criteria where there
    # was at least one characteristic that is in demand/excess. The strictness needs to be between 1 and the total number of masks if one is to make nCk
    # combinations of masks, where n is the number of masks, and k is the strictness.
    strictness = min(strictness, len(masks))
    dfs = []
    # Create the combinations and iterate over them. If the criteria are e.g. gender, age, and education, and the strictness is 2, the combinations will be the
    # masks
    # (gender, age)
    # (gender, education)
    # (age,    education)
    for comb in combinations(masks, strictness):
        # [1] Create a tuple of the masks of all the criteria in the combination
        # [2] Unpack the tuple and group all elements together by index (the first elements of each mask are toether, as well as the second elements, etc.)
        # [3] Evaluate to True at indices where _all_ masks are True. False everywhere else. This creates a new mask with the same length as the previous ones and
        # as the population DataFrame
        # [4] Apply the new mask to the population to get all the people from the population that has _any_ relevant characteristic from _all_ the criteria in the
        # current combination
        #      |--[4]---...|-----[3]---... [2]--|------------[1]-----------|
        vols = population[[all(x) for x in zip(*(masks[key] for key in comb))]]
        dfs.append(vols)
    return dfs


def get_distance(overview: dict[pd.DataFrame]) -> int:
    """A function to calculate the total deaition of a sample from the criteria.

    Args:
        overview (dict[pd.DataFrame]): A dict of DataFrames with criteria as keys. See the output of /get_overview/.

    Returns:
        int: Sum of the absolute values of all the "difference" columns.
    """

    s = 0
    for df in overview.values():
        s += df["difference"].apply(abs).sum()
    return s


def validate_crtiera(criteria: dict) -> bool:
    """Function to validate that all criteria demand the same total number of people.

    Args:
        criteria (dict): Dict describing the desired distributon of each criterium.

    Returns:
        bool: Whether the criteria pass the test or not.
    """
    totals = np.zeros(len(criteria), dtype=int)
    for i, value in enumerate(criteria.values()):
        totals[i] = sum([number for number in value["values"].values()])
    return np.all(totals == totals[0])


def main_iteration(
    sample: pd.DataFrame,
    volunteers: pd.DataFrame,
    overview: dict[pd.DataFrame],
    criteria: dict[str],
) -> pd.DataFrame:
    """Main iteration loop. Tries to find the people in a sample with the most characteristics that are in excess and replace them with people not yet in the lotting with the most characteristics that are in demand.

    Args:
        sample (pd.DataFrame): Satring sample of people that have been lotted.
        volunteers (pd.DataFrame): Full dataframe of all volunteers, both those who have been lotted, and those who have not.
        overview (dict[pd.DataFrame]): Overview describing the distribution of characteristics of each criteria, compared with the desired distribution, See ouput of /get_overview/
        criteria (dict[str]): Criteria describing the desired distribution of characteristics.

    Returns:
        pd.DataFrame: A new sample with one person wih charateristics in excess swapped with another person with characteristics in demand.
    """
    # Set the initial strictness to be equal to the length of the overview. (Also equal to the total number of criteria)
    strictness = len(overview)
    # A mask of which people in the /volunteers/ DataFrame are also in the /sample/ DataFrame. These are the people who have already been lotted.
    isin_mask = volunteers["Navn"].isin(sample["Navn"])
    # Inital distance
    distance = get_distance(overview)

    # Loop over ever decreasing strictnesses until a swappable match is found.
    while strictness >= 1:
        # Get list of dicts with DataFrames of people who have not been lotted but have characteristics that are in demand
        in_demand = excess_or_demand(overview, volunteers[~isin_mask], strictness)
        # Get list of dicts with DataFrames of people who have been lotted, but who have characeristics that are in excess
        in_excess = excess_or_demand(overview, sample, strictness, demand=False)

        # Iterate over the DataFrames. The people in demnd will have characteristics in demand in the same criteria as the people in excess have characteristics
        # in excess. E.g. if the criteria are (gender, education), and we are in demand of people who are female, and people who have a university education, but have an excess of people who are male, and people who have high school education, the DataFrames will contain such individuals respectively.
        for dem, exc in zip(in_demand, in_excess):
            # Randomly sort the people in demand and iterate over their indices
            for i in dem.sample(len(dem)).index:
                # Access the current volunteer in demand
                vol = dem.loc[[i]]

                # Iterate over the volunteers in excess in random order
                for j in exc.sample(len(exc)).index:
                    ex_vol = exc.loc[[j]]

                    # Create a temporary sample where the current person in excess is swapped for the current person in demand.
                    temp_sample = pd.concat(
                        [sample.drop(ex_vol.index[0]), vol], ignore_index=True
                    )
                    # If the new temporary sample has a lower distance than the original one, return is as the new sample.
                    if get_distance(get_overview(temp_sample, criteria)) <= distance:
                        return temp_sample
        # If after looping through all combinations of criteria there has not been found any suitable pairs of volunteers to swap, reduce the strictness and
        # repeat.
        strictness -= 1
    # If after reducing the strictness to 1 no suitable matches to swap have been found, return the original sample.
    return sample


class CriteriaError(Exception):
    pass


def main(
    starting_sample: pd.DataFrame, volunteers: pd.DataFrame, criteria: dict
) -> pd.DataFrame:
    """Main function. Takes a starting sample of volunteers and successively swaps those with characteristics in excess for people with characteristics in demand.

    Args:
        starting_sample (pd.DataFrame): Starting sample of lotted participants. My be completely random or pre-lotted.
        volunteers (pd.DataFrame): Dataframe of all volunteers. Both those who have been lotted, and those who have not.
        criteria (dict): The criteria describing the desired distribution of characteristics.

    Raises:
        CriteriaError: If not all criteria demand the same total number of people.

    Returns:
        pd.DataFrame: Final optimal lotting.
    """
    if not validate_crtiera(criteria=criteria):
        raise CriteriaError("Not all criteria demand the same total number of people")

    sample = starting_sample.copy()
    overview = get_overview(sample, criteria)
    old_distance = np.inf
    distance = get_distance(overview)

    # Iterate until a perfect lotting is found.
    while distance > 0:
        # Swap one person in excess with one person in demand
        sample = main_iteration(sample, volunteers, overview, criteria)

        # Get new overview and distance
        overview = get_overview(sample, criteria)
        old_distance = distance
        distance = get_distance(overview)

        # If distance does not improve, quit loop.
        if old_distance == distance:
            print(f"Distance not improving beyond {int(distance)}. Exiting.")
            break
    # Return final optimal sample.
    return sample


if __name__ == "__main__":
    parser = ArgumentParser("Automatic Sortition")

    parser.add_argument("-v", "--volunteers", default="data/volunteers.xlsx")
    parser.add_argument("-c", "--criteria", default="criteria/criteria.json")
    parser.add_argument("-o", "--output", default="results/output.xlsx")

    args = parser.parse_args()

    output = Path(args.output)

    volunteers = pd.read_excel(args.volunteers)
    criteria = json.load(Path(args.criteria).open())
    sample = volunteers.sample(
        sum(criteria[list(criteria.keys())[0]]["values"].values())
    )

    lotting = main(starting_sample=sample, volunteers=volunteers, criteria=criteria)

    overview = get_overview(lotting, criteria)

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        lotting.to_excel(writer, sheet_name="lotting", index=False)
        for criterium, view in overview.items():
            pd.DataFrame(view).to_excel(writer, sheet_name=criterium)
