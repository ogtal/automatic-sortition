# Automatic sortition

This project is meant to be used for doing a lottery of participants of citizen's asseblies. The program is given a list of potential participants with their corresponding characeristics, and a set of criteria for the desired distribution of characteristics. The program then finds the optimal allottment of participants so that the combination of characteristics best match the desired distribution.

## Installing

The project can be installed using a tool like Poetry. All of the projects dependencies are listed in the file [pyproject.toml](./pyproject.toml) and are automatically installed with the project. Simply run

```shell
$ git clone git@github.com:ogtal/automatic-sortition.git
$ cd automatic-sortition
$ poetry install
```


## Running

To run the program, simply run 

```
$ poetry shell
$ python3 automatic_sortition.py
```

The program expects two files to be present to run: An Excel-file containing the population of volunteers from which the sample is to be drawn, and a JSON-file describing the desired distribution of characteristics in the final sample. If no arguments are given the program will look for these in [./data/volunteers.xlsx](./data/volunteers.xlsx) and [./criteria/criteria.json](./criteria/criteria.json) respectively. The results will be saved to [./results/output.xlsx](./results/output.xlsx) if not overruled by parameters.

Possible paramters are:

`-v <filepath>` or `--volunteers <filepath>` giving the path to where the volunteers file can be found

`-c` `<filepath>` or `--criteria <filepath>` giving the path to where the criteria file can be found

`-o <filepath>` or `--output <filepath>` giving the path to where the result file should be saved

Given a sufficiently large set of volunteers, a perfect solution where all crtieria are fulfilled perfectly should often exist, and the program usually finds it. It might on some runs end up in a local minimum where swapping one person from the sample for another does not yield better fulfillment, even if a better solution exists. If so, one can simply rerun the program, as this will produce a new result due to the stochastic nature of the algorithm.

## License

The project is licensed under the GNU Affero General Public License version 3. The license can be found in the [LICENSE](./LICENSE) file, or at [https://www.gnu.org/licenses/agpl-3.0.html](https://www.gnu.org/licenses/agpl-3.0.html).


## Attributions

This project is heavily inspired by [swidish-sortition](https://github.com/digidemlab/swedish-sortition) by [@PierreMesure](https://github.com/PierreMesure) at [@digidemlab](https://github.com/digidemlab). A huge thanks goes to them for their hard work.
