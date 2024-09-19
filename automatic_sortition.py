import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from xlsxwriter import Workbook

"""
Term definitions:
A *characteristic* is an attribute that a person can have, e.g. age, locality, education level, or gender.
A *group* in this context refers to the value of a given characterisitc attribute, e.g. having the gender "man", the locality "Oslo", or the age "16-25".
"""


def get_want(criteria: dict) -> pl.DataFrame:
    """A function for creating a pl.DataFrame showing how many people from each group for each characteristic is wanted in the final sample.

    Args:
        criteria (dict): A dictionary containing the criterium data.

    Returns:
        pl.DataFrame: A dataframe where each row corresponds to a group, showing which characteristic it belongs to, the number of people wanted belonging to the given group, and the priority of the characteristic.
    """
    char = []
    group = []
    number = []
    priority = []
    ## Iterate over the criteria
    for crit, dic in criteria.items():
        char += (
            [crit] * len(dic["values"])
        )  # Add to the list of cahracteristics. E.g. ["Age", "Age", "Age", "Age", "Age"]
        group += list(
            dic["values"].keys()
        )  # Add to the list of groups. E.g. ["16-25", "26-35", "36-45", "46-59", "60+"]
        number += list(
            dic["values"].values()
        )  # Add to the list of number of people wanted. E.g. [2, 2, 1, 1, 2]
        ## Add to the list of priority numbers. If no priority is given, set the priority to 1. E.g. [2, 2, 2, 2, 2]
        priority += [dic["priority"] if "priority" in dic.keys() else 1] * len(
            dic["values"]
        )
    ## After adding to the lists for all characteristics, create a dataframe with the lists as columns and return it.
    return pl.DataFrame(
        {
            "characteristic": char,
            "group": group,
            "priority": pl.Series(priority).cast(pl.Int64),
            "want": pl.Series(number).cast(pl.Int64),
        }
    )


def get_have(sample: pl.DataFrame) -> pl.DataFrame:
    """Get a pl.DataFrame showing how many people in a given sample belong to each group in each caharacteristic.

    Args:
        sample (pl.DataFrame): The sample to be described.

    Returns:
        pl.DataFrame: A discription of the distribution of characteristics in the sample
    """
    dfs = []
    ## Iterate over the cahracteristics and get a description of the distribution in the sample of each
    for char in list(criteria.keys()):
        dfs.append(
            sample.group_by(char)
            .len()
            .with_columns(pl.lit(char).alias("characteristic"))
            .rename({char: "group", "len": "have"})
        )
    return pl.concat(dfs).select(
        "characteristic", "group", pl.col("have").cast(pl.Int64)
    )


def get_overview(sample: pl.DataFrame, want: pl.DataFrame) -> pl.DataFrame:
    """Get an overview of the wanted distribution of characteristics in the sample, the actual distribution, and the difference between these.

    Args:
        sample (pl.DataFrame): The sample to be described.
        want (pl.DataFrame): The wanted distribution of characteristics.

    Returns:
        pl.DataFrame: A dataframe with five columns "characteristic", "group", "priority", "have", "want", and "diff"
    """
    have = get_have(sample)
    return (
        have.join(want, on=("characteristic", "group"), how="full", coalesce=True)
        .sort("characteristic", "priority", nulls_last=True)
        .with_columns(
            pl.col("have", "want").fill_null(0),
            pl.col("priority").fill_null(strategy="forward"),
        )
        .with_columns((pl.col("have") - pl.col("want")).alias("diff"))
        .sort("priority", "characteristic", "group")
        .select("characteristic", "group", "priority", "have", "want", "diff")
    )


def get_reserves(
    sample: pl.DataFrame, volunteers: pl.DataFrame, criteria: dict
) -> pl.DataFrame:
    """Append a column to sample indicating the number of volunteers not yet in the sample who have the exact same profile as the ones in the sample.

    Args:
        sample (pl.DataFrame): The sample to find reserves for.
        volunteers (pl.DataFrame): All volunteers.
        criteria (dict): Description of desired distribution of characteristics.
    """
    ## Get a list of all the characteristics, e.g. ["Age", "Gender, Locality", "Education"]
    chars = list(criteria.keys())
    ## Filter out the volunteers who are already in the sample
    available_volunteers = volunteers.filter(
        ~pl.col("index").is_in(sample.get_column("index"))
    )
    ## Count how many times each unique profile occurs among the available volunteers
    volunteers_count = available_volunteers.group_by(chars).agg(
        pl.len().alias("Reserves")
    )
    ## Join the people in the sample with the counts of each profile on all relevant characteristics, giving the desired new column with number of reserves. If no reserves exists, the joing yields None for that profile. This is the nfilled with zeros.
    return sample.join(volunteers_count, on=chars, how="left", coalesce=True).fill_null(0)


def excess_and_demand(
    sample: pl.DataFrame, want: pl.DataFrame, relevant_characteristic: str
) -> tuple[pl.Series, pl.Series]:
    """Get two pl.Series: One with the groups that are in excess and one with groups that a re in demand for a given characteristic. E.g. if the given characteristic is "Locality" excess might be pl.Series(["Oslo"]), while demand might be pl.Series(["Bergen", "Trondheim"]).

    Args:
        sample (pl.DataFrame): The sample to analyze
        want (pl.DataFrame): Dataframe describing the desired distribution of characteristics
        relevant_characteristic (str): The characteristic we want to analyze

    Returns:
        tuple[pl.Series, pl.Series]: The groups that are in excess and demand respectively.
    """
    overview = get_overview(sample, want)
    excess_temp = (
        overview.filter(
            (pl.col("characteristic").eq(relevant_characteristic), pl.col("diff") > 0)
        )
        .group_by("characteristic")
        .agg("group")
        .get_column("group")
    )
    demand_temp = (
        overview.filter(
            (pl.col("characteristic").eq(relevant_characteristic), pl.col("diff") < 0)
        )
        .group_by("characteristic")
        .agg("group")
        .get_column("group")
    )
    excess = excess_temp.item() if len(excess_temp) > 0 else pl.Series([])
    demand = demand_temp.item() if len(demand_temp) > 0 else pl.Series([])

    return excess, demand


def get_distance(
    sample: Optional[pl.DataFrame] = None,
    want: Optional[pl.DataFrame] = None,
    overview: Optional[pl.DataFrame] = None,
) -> int:
    """A simple helper to calculate the distance between the current and wanted distribution of charachteristics.

    Args:
        sample (pl.DataFrame): The sample to find the distance from.
        want (pl.DataFrame): The desired distribution.
        overview (pl.DataFrame): Optionally provide just the overview.

    Returns:
        int: The distance as an integer
    """
    assert (sample is not None and want is not None) or (
        overview is not None
    ), "Either both sample and want, or overview, must be provided."
    if overview is None:
    overview = get_overview(sample, want)
    return overview.select(pl.col("diff").abs()).sum().item() // 2


def iteration(
    sample: pl.DataFrame, volunteers: pl.DataFrame, want: pl.DataFrame
) -> pl.DataFrame:
    """A single iteration swapping a maximum of one person for each cahracterisitc

    Args:
        sample (pl.DataFrame): The starting sample
        volunteers (pl.DataFrame): A dataframe containing all volunteers: Both those already in the sample and those not.
        want (pl.DataFrame): A dataframe describing the desired distribution of characteristics.

    Returns:
        pl.DataFrame: A new sample of the same length as the input sample, but with some participants swapped.
    """

    ## Get a list of the characteristics in prioritized order. E.g. ["Gender", "Age", "Locality", "Education"]
    chars = (
        want.sort(pl.col("priority"))  # Sort by priority
        .get_column(
            "characteristic"
        )  # Get only the column with the characteristic names
        .unique(
            maintain_order=True
        )  # Maintain only one row of each characteristic name, but keep the prioritized order
        .to_list()  # Convert to list
    )

    ## Iterate over the cahracteristics from highest to lowest priority
    for i, char in enumerate(chars):
        ## Get the groups that are in excess and demand respectively for the current characteristic. E.g. if char is "Locatity" and sample contains fewer people from Oslo, but more people from Bergen and Trondheim, than we want, excess and demand will be pl.Series(["Bergen", "Trondheim"]) and pl.Series(["Oslo"]) respectively.
        excess, demand = excess_and_demand(sample, want, char)

        ## Get the IDs of the people that are in the sample who belong to a group that is in excess for the given characteristic. E.g. if excess is pl.Series(["Bergen", "Trondheim"]), excess_ids will be a pl.Series containing the IDs of all the people in sample who are from Bergen or Trondheim
        excess_ids = (
            sample.filter(
                pl.col(char).is_in(excess)
            )  # Get the rows from sample where the value in the column corresponding to the current characteristic is in the list of groups in excess
            .sample(fraction=1)  # Randomly shuffle the order
            .get_column("index")  # Get only the index column
        )
        ## If there are no people in excess, continue to the next cahracteristic with lower priority, e.g. go from "Locality" to "Education"
        if len(excess_ids) == 0:
            continue

        ## Iterate over the poeple in excess and swap them out for a person who belongs to a group that is in demand for the current characteristic, but belongs to all the same groups as the person in excess for all the characteristics with higher priority than the current. E.g. if demand is pl.Series(["Oslo"]) and excess is pl.Series(["Bergen", "Trondheim"]), and excess_person belong to the following groups for all characrterisitcs {"Gender": "Man", "Age": "16-25", "Locality": "Bergen", "Education": "Primary school"}, we can swap them out with someone not yet in the sample who belong to the following groups {"Gender": "Man", "Age": "16-25", "Locality": "Oslo", "Education": "High school"}. Note that the new person belongs to the same groups as excess_person for the characteristics of higher priority ("Gebder" and "Age") than the current characteristic ("Locality"). They also belong to a group that is in demand for the current characteristic. We can also see that their group belonging in "Education" has changed, but this does not matter since the cahracteristic "Education" has lower priority than the current characteristic "Locality".
        for excess_id in excess_ids:
            ## Extract the person with the current excess_id
            excess_person = sample.filter(pl.col("index").eq(excess_id))
            ## Extract the poeple that are in demand
            demand_people = volunteers.filter(  # Extract the volunteers who...
                (
                    pl.col(char).is_in(
                        demand
                    ),  # ... belong to a group that is in demand for the current characteristic...
                    ~pl.col("index").is_in(
                        sample.get_column("index")
                    ),  # ... and are not already in the sample.
                )
            )
            ## If there are any characteristics that have higher priority than the current characteristic, remove people from the list of people in demand if they do not beliong to the same groups as excess_person for these higher priority characteristics.
            if len(chars[:i]) > 0:
                demand_people = demand_people.join(  # Return the people in demand who,...
                    excess_person,  # when compared to the person in excess...
                    on=chars[:i],  # on the characteristics with higher priority,...
                    how="semi",  # belong to the same groups. (See https://docs.pola.rs/user-guide/transformations/joins/#join-strategies for how a semi-join works)
                )
            ## If there are no people in demand, continue to the next person in excess
            if len(demand_people) == 0:
                continue
            ## Finally remove the person in excess and insert a random person in demand
            sample = pl.concat(
                [
                    sample.filter(
                        ~pl.col("index").eq(excess_id)
                    ),  # The original sample minus the person in excess
                    demand_people.sample(1),  # A single random person in demand
                ]
            )
            ## Continue to the next cahracteristic with lower priority
            break
    ## Return the swapped sample
    return sample


def main(volunteers: pl.DataFrame, criteria: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    sample = volunteers.sample(
        sum(criteria[list(criteria.keys())[0]]["values"].values())
    )
    want = get_want(criteria)
    old_distance = np.inf
    distance = get_distance(sample=sample, want=want)

    while distance > 0:
        sample = iteration(sample, volunteers, want)
        old_distance = distance
        distance = get_distance(sample=sample, want=want)
        if old_distance == distance:
            sample = iteration(sample, volunteers, want)
            break
    sample = get_reserves(sample=sample, volunteers=volunteers, criteria=criteria)
    return sample.sort("index"), get_overview(sample, want)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Automatic Sortition",
        description="Create a random sortition of people that tries to fulfill a given set of criteria as well as possible.",
    )

    parser.add_argument(
        "-v",
        "--volunteers",
        help="Location of Excel-file containing rows of volunteers.",
        type=Path,
        default="data/volunteers.xlsx",
    )
    parser.add_argument(
        "-c",
        "--criteria",
        help="Location of JSON-file containing the desired distribution of characteristics.",
        type=Path,
        default="criteria/criteria.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Location of output file.",
        type=Path,
        default="results/output.xlsx",
    )

    args = parser.parse_args()

    output = Path(args.output)

    volunteers = pl.read_excel(args.volunteers).with_row_index()
    criteria = json.load(args.criteria.open())

    lotting, overview = main(volunteers=volunteers, criteria=criteria)

    reserves_count = lotting.get_column("Reserves").value_counts()
    no_reserves = (
        reserves_count.filter(pl.col("Reserves").eq(0)).get_column("count").sum()
    )
    one_reserve = (
        reserves_count.filter(pl.col("Reserves").eq(1)).get_column("count").sum()
    )
    more_than_two_reserves = (
        reserves_count.filter(pl.col("Reserves").ge(2)).get_column("count").sum()
    )
    print(
        f"Distance: {int(get_distance(overview=overview))}\n0 reserves:  {no_reserves}\n1 reserve:   {one_reserve}\n>2 reserves: {more_than_two_reserves}\n"
    )

    with Workbook(filename=output) as workbook:
        lotting.select(pl.exclude("index")).write_excel(
            workbook=workbook, worksheet="lotting"
        )
        for char in overview.get_column("characteristic").unique():
            overview.filter(pl.col("characteristic").eq(char)).select(
                "group", "have", "want", "diff"
            ).write_excel(workbook=workbook, worksheet=char)
