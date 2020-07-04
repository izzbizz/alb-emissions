import pandas as pd
from sklearn.model_selection import train_test_split

class Preparer():
    '''prepare the dataset for training
       modify and remove columns, delete rows with nans or nonsensical values
       returns training set, test set, evaluation set'''
    def __init__(self, datapath, filename):
        self.df = pd.read_csv(datapath + filename, parse_dates=['TEST_SDATE', 'TEST_EDATE'], lineterminator='\n', low_memory=False)
        
    def _create_day_column(self):
        self.df['DAY'] = self.df.TEST_SDATE.apply(lambda x: x.date())

    def _create_age_column(self):
        '''add one year to model year to avoid age 0'''
        self.df['AGE'] = pd.to_datetime(self.df["TEST_EDATE"]).dt.year - self.df["MODEL_YEAR"]
        self.df['AGE'] += 1

    def _create_target_variable(self):
        '''target variable is the cars future test result
           remove rows without future text result'''
        self.df['TARGET'] = self.df.groupby('VIN').OVERALL_RESULT.shift(-1)
        self.df = self.df[self.df.TARGET.notna()]

    def _remove_daily_duplicates(self):
        '''remove duplicate tests on the same day'''
        self.df.drop_duplicates(['VIN', 'DAY'], keep='first', inplace=True)

    def _reduce_results(self):
        '''remove irrelevant test results'''
        self.df = self.df[self.df.TARGET.isin(['P', 'F'])]

    def _remove_rows(self):
        '''remove nonsensical odometer and nan vehicle type'''
        self.df = self.df[(~self.df.ODOMETER.isin([8888888, 888888, 88888, 100000, 0, 1, 10000])) \
                    & (~self.df.GVW_TYPE.isna())]

    def _select_features(self):
        self.df = self.df[['ODOMETER', 'ESC', 'STATION_NUM', 'GVW_TYPE', 'E_HIGH_CO_RESULT', 'E_IDLE_CO_RESULT', \
                            'E_HIGH_HC_RESULT', 'E_IDLE_HC_RESULT', 'V_SMOKE1', 'V_SMOKE2', 'V_CAT', 'OVERALL_RESULT', \
                            'TARGET', 'AGE', 'KOEO_RESULT', 'KOER_RESULT', 'E_TEST_SEQUENCE', 'OBD_RESULT', 'SOFTWARE_VERSION']]

    def _clean_df(self):
        self._create_day_column()
        self._create_age_column()
        self._remove_daily_duplicates()
        self._create_target_variable()
        self._reduce_results()
        self._remove_rows()
        self._select_features()

    def _stringify_columns(self):
        self.df['ESC'] = self.df.ESC.astype(str)
        self.df['GVW_TYPE'] = self.df.GVW_TYPE.astype(str)
        self.df['SOFTWARE_VERSION'] = self.df.SOFTWARE_VERSION.astype(str)

    def _dummify_categoricals(self):
        dummies = pd.get_dummies(self.df[['ESC', 'STATION_NUM', 'GVW_TYPE', 'E_HIGH_CO_RESULT', 'E_IDLE_CO_RESULT', 'E_HIGH_HC_RESULT',\
                                           'E_IDLE_HC_RESULT', 'V_SMOKE1', 'V_SMOKE2', 'V_CAT', 'OVERALL_RESULT', 'KOEO_RESULT', \
                                            'KOER_RESULT', 'OBD_RESULT', 'SOFTWARE_VERSION']], drop_first=True, dummy_na=False)
        self.df = pd.concat([self.df[['ODOMETER', 'AGE', 'E_TEST_SEQUENCE', 'TARGET']], dummies], axis=1)

    def make_train_test_data(self):
        self._clean_df()
        self._stringify_columns()
        self._dummify_categoricals()
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop('TARGET', axis=1), self.df.TARGET, test_size=.3, random_state=33)        
        X_test, X_eval, y_test, y_eval = train_test_split(X_test, y_test, test_size=.33, random_state=0)
        
        return X_train, X_test, X_eval, y_train, y_test, y_eval 

        
