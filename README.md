# alb-emissions

This repository is an outcome of the Berlin Data Science Lab organized by the Women in Data Science and Machine Learning group. We were four women working on the problem of forecasting car emissions. The project works with [Albuquerque's open data set of car emission check results](http://data.cabq.gov/airquality/vehicleemissions/). The pipeline for downloading and preprocessing the files, follow the instructions on the [Berlin DS Lab site](https://github.com/wimlds/berlin-ds-lab). Using the preprocessed files, I did an in-depth exploratory data analysis that I present here in a Jupyter notebook (see [this link](https://github.com/izzbizz/alb-emissions/blob/master/eda/general_eda.ipynb) for part 1 and [this one](https://github.com/izzbizz/alb-emissions/blob/master/eda/emissions_eda.ipynb) for part 2 of the EDA). The outcome in turn fed into my forecasting model. I trained an Xgboost classifier that predicts whether a car will fail an emission test in the upcoming two years.

## To use the model:
1. Clone the repository

`git clone https://github.com/izzbizz/alb-emissions.git`

2. Go to project

`cd alb-emissions`

3. Create a virtual environment using conda

`conda create --name alb python==3.7`

4. Activate virtual enviroment

`conda activate alb`

5. Install the packages from `requirements.txt`

`pip install -r requirements.txt`
