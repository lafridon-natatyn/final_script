import datetime

import pandas as pd
import dill

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


print('x start')
df1 = pd.read_csv('data/ga_hits.csv')
print(list(df1))
df2 = pd.read_csv('data/ga_sessions.csv', low_memory=False)
df1['target'] = df1.apply(
        lambda x: 1 if x.event_action in ['sub_car_claim_click', 'sub_car_claim_submit_click',
                                              'sub_open_dialog_click', 'sub_custom_question_submit_click',
                                              'sub_call_number_click', 'sub_callback_submit_click',
                                              'sub_submit_success', 'sub_car_request_submit_click'] else 0, axis=1)
print('target ready')
df = pd.merge(left=df1.groupby('session_id')['target'].max(), right=df2, on='session_id', how='inner')
print('merged')
print(list(df))



def main():
    import dill
    import pandas as pd
    global pipe, model, X, y

    # df1 = pd.read_csv('data/ga_hits.csv')
    # df2 = pd.read_csv('data/ga_sessions.csv', low_memory=False)

    def filter_data(df):
        print('filter start')
        df = df.copy()
        columns_to_drop = [
            # # 'hit_date',
            # # 'hit_number',
            # # # 'hit_type',
            # # # 'hit_page_path',
            # # 'event_category',
            # 'event_action',
            'device_model',
            'utm_keyword',
            'device_screen_resolution',
            'client_id',
            'session_id',
            'visit_number',
            'visit_time',
            'visit_date',
            'utm_campaign',
            'utm_medium',
            'device_os',
            'device_brand',
            'device_browser',
            'geo_country',
            'utm_source',
            'utm_adcontent',
            'geo_city',
            'utm_campaign',

        ]
        df = df.drop(columns_to_drop, axis=1)
        print(list(df))
        print('filtred')
        return df

    # def merge_df():
    #     import pandas as pd
    #     print('x start')
    #     df1 = pd.read_csv('data/ga_hits.csv')
    #     df2 = pd.read_csv('data/ga_sessions.csv', low_memory=False)
    #     df1['target'] = df1.apply(
    #         lambda x: 1 if x.event_action in ['sub_car_claim_click', 'sub_car_claim_submit_click',
    #                                           'sub_open_dialog_click', 'sub_custom_question_submit_click',
    #                                           'sub_call_number_click', 'sub_callback_submit_click',
    #                                           'sub_submit_success', 'sub_car_request_submit_click'] else 0, axis=1)
    #     print('target ready')
    #     df = pd.merge(left=df1.groupby('session_id')['target'].max(), right=df2, on='session_id', how='inner')
    #     print('merged')
    #     return df

    # def x_df(df):
    #     print('x start')
    #     X = df.drop(['target'], axis=1)
    #     print('x ready')
    #     return X
    #
    # def y_df(df):
    #     print('y start')
    #     y = df['target']
    #     print('y ready')
    #     return y

    def new_features(df):

        # df1 = df1.copy()
        # print('1 copied')
        # df2 = df2.copy()
        print('nf start')
        import pandas as pd
        # df1['target'] = df1.apply(
        #     lambda x: 1 if x.event_action in ['sub_car_claim_click', 'sub_car_claim_submit_click',
        #                                       'sub_open_dialog_click', 'sub_custom_question_submit_click',
        #                                       'sub_call_number_click', 'sub_callback_submit_click',
        #                                       'sub_submit_success', 'sub_car_request_submit_click'] else 0, axis=1)
        # print('target ready')
        df['utm_adcontent_clean'] = df.apply(
            lambda x: x.utm_adcontent if x.utm_adcontent in ['JNHcPlZPxEMWDnRiyoBf', 'vCIpmpaGBnIQhyYNkXqp',
                                                             'xhoenQgDQsgfEPYNPwKO', 'PkybGvWbaqORmxjNunqZ',
                                                             'LLfCasrxQzJIyuldcuWy', 'TuyPWsGQruPMpKvRxeBF',
                                                             'UxrnyMlRBSOhOjytXnMG', 'dUuXlWzvmhDSyclWRhNP',
                                                             'yYdBRbPmBMUZHXwqGxNx', 'WYLajZgbUhGimwBKDZUH',
                                                             'SOkCdPxfUcZUzzOdgGES', 'AdeErYgVTbRcAWtHrMHq',
                                                             'nNqUcgFgcqQbTVSvgaHr',
                                                             'aYAcKhelKzYpXrRYknSP'] else 'other', axis=1)
        print('utm_adcontent_clean ready')
        df['utm_source_clean'] = df.apply(
            lambda x: x.utm_source if x.utm_source in ['ZpYIoDJMcFzVoPFsHGJL', 'fDLlAcSmythWSCVMvqvL',
                                                       'kjsLglQLzykiRbcDiGcD', 'MvfHsxITijuriZxsqZqt',
                                                       'BHcvLfOaCWvWTykYqHVe', 'bByPQxmDaMXgpHeypKSM',
                                                       'QxAxdyPLuQMEcrdZWdWb', 'aXQzDWsJuGXeBXexNHjc',
                                                       'jaSOmLICuBzCFqHfBdRg', 'RmEBuqrriAfAVsLQQmhk',
                                                       'vFcAhRxLfOWKhvxjELkx', 'PlbkrSYoHuZBWfYjYnfw',
                                                       'hTjLvqNxGggkGnxSCaTm', 'gDBGzjFKYabGgSPZvrDH'] else 'other',
            axis=1)
        # print('utm_source_clean_clean ready')
        df['geo_city_clean'] = df.apply(
            lambda x: x.geo_city if x.geo_city in ['Moscow', 'Saint Petersburg', 'Yekaterinburg', 'Krasnodar', 'Kazan',
                                                   'Samara', 'Nizhny Novgorod', 'Samara', 'Nizhny Novgorod', 'Ufa',
                                                   'Novosibirsk', 'Krasnoyarsk', 'Chelyabinsk', 'Tula', 'Voronezh',
                                                   'Rostov-on-Don', 'Irkutsk', 'Grozny', 'Balashikha',
                                                   'Vladivostok'] else 'other', axis=1)
        # print('geo_city_clean_clean ready')
        df['device_browser_clean'] = df.apply(
            lambda x: x.device_browser if x.device_browser in ['Chrome', 'Safari', 'YaBrowser', 'Safari (in-app)',
                                                               'Android Webview', 'Samsung Internet', 'Opera',
                                                               'Firefox', 'Edge'] else 'other', axis=1)
        # print('device_browser ready')
        df['device_brand_clean'] = df.apply(
            lambda x: x.device_brand if x.device_brand in ['ZTE', 'Sony', 'Nokia', 'Asus', 'OnePlus', 'Vivo', 'OPPO',
                                                           'Realme', 'Huawei', 'Xiaomi', 'Samsung',
                                                           'Apple'] else 'other', axis=1)
        # print('device_brand_clean ready')
        df['geo_country_clean'] = df.apply(
            lambda x: x.geo_country if x.geo_country in ['Russia', 'United States', 'Ukraine', 'Ireland', 'Belarus',
                                                         'Sweden', 'Kazakhstan', 'Germany', 'Turkey', 'Netherlands',
                                                         'Uzbekistan', 'United Kingdom'] else 'other', axis=1)
        # print('geo_country_clean ready')
        df['utm_medium_clean'] = df.apply(
            lambda x: x.utm_medium if x.utm_medium in ['smartbanner', 'blogger_channel', 'cpv', 'stories', 'push',
                                                       'email', 'organic', 'referral', 'cpm', 'other', 'cpc', 'banner',
                                                       'outlook', 'smm', 'post', 'app', 'tg', 'cpa',
                                                       'blogger_stories'] else 'other', axis=1)
        # print('utm_medium_clean ready')

        # df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
        # df['age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))

        return df



    # df = merge_df()
    X = df.drop(['target'], axis=1)
    y = df['target']


    categorical = make_column_selector(dtype_include=['object','int64', 'float64'])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='other')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    transformer = Pipeline(steps=[
        # ('merge', FunctionTransformer(merge_df)),
        ('new_features', FunctionTransformer(new_features)),
        ('filter', FunctionTransformer(filter_data)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical),
    ])

    model = RandomForestClassifier(random_state=42)

    best_score = .0
    best_pipe = None



    pipe = Pipeline(steps=[
        ('transformer', transformer),
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    # score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc_ovr')

    # X = df.drop(['target'], axis=1)
    # y = df['target']
    # print(X['session_id'].head())
    pipe.fit(X,y)
    score = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
    # print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}'
    print(f'model: {type(model).__name__}, roc_auc: {score:.4f}')


    with open('sberautopodpiska.pkl', 'wb') as output_file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'Target action prediction model',
                'author': 'Ilya Demchenko',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(pipe.named_steps["classifier"]).__name__,
                'accuracy': score
            }
        }, output_file)


if __name__ == '__main__':
    main()
# Как установить FastAPI
# Открыть терминал, ввести команду conda activate.
# Посмотреть список всех окружений в системе командой conda env list.
# Поменять окружение на loan_service командой source activate loan_service.
# Установить пакеты при помощи команды pip install "fastapi[all]". Слово all в квадратных скобках после имени пакета fastapi указывает, что нужно установить все пакеты и программы, связанные с fastapi.
# Вернуться в файл main.py в редакторе кода.
# Импортировать FastAPI из пакета fastapi, создать объект класса FastAPI.