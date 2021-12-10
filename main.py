import requests
import pandas as pd
import numpy as np
import datetime
#from datetime import date, timedelta
import time
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
import itertools
from fbprophet.diagnostics import cross_validation
#import xlsxwriter
from scipy.optimize import curve_fit
import pandahouse as ph

#PREDICT_FROM = '2021-08-01'
PREDICT_FROM = datetime.datetime.today().strftime('%Y-%m-%d')

def delete():
    sql="""
    ALTER TABLE pulse.stats_prediction_replicated on cluster tableau DELETE WHERE toYYYYMMDD(predict_date) = toYYYYMMDD(toDate('{}'))
    """.format(PREDICT_FROM)
    connection = {'host': ' http://proxy.surfy.ru:8125/?user=pulse_ml_rw&password=Tee7ohyae4aiVuac',
                  'database': 'pulse'}

    df =  ph.read_clickhouse(sql,
                            connection=connection)
    return df

# получение данных
def get_redash_result(query, readsh_params):
    api_key = ''

    redash_url = 'https://redash-ml.surfy.ru'
    headers = {'Authorization': 'Key {}'.format(api_key)}

    response = requests.post(
        redash_url+'/api/queries/{}/refresh'.format(query),
        headers=headers,
        params=readsh_params
    )
    print('responseresponse',response)
    while response.json()['job']['status'] not in (3,4):
        s = requests.Session()
        s.headers.update({'Authorization': 'Key {}'.format(api_key)})
        response = s.get('{}/api/jobs/{}'.format(redash_url, response.json()['job']['id']))
        job = response.json()['job']
        time.sleep(1)

        if job['status'] == 3:
            response = s.get(redash_url+'/api/query_results/{}'.format(job['query_result_id']))
            return response.json()['query_result']['data']['rows']
    raise Exception('No data recieved! ' + str(response.json()))

def add_regressor(mon):
    date = pd.to_datetime(mon)
    if date>= pd.to_datetime('2021-04-06') and date <= pd.to_datetime('2021-06-09'):
        return 1
    else:
        return 0
def add_regressor1(mon):
    date = pd.to_datetime(mon)
    if date>= pd.to_datetime('2020-08-24') and date <= pd.to_datetime('2020-11-18'):
    #if date <= pd.to_datetime('2020-12-01'):
        return date.dayofyear
    else:
        return 0
def add_regressor2(mon):
    date = pd.to_datetime(mon)
    if date>= pd.to_datetime('2021-04-06') and date <= pd.to_datetime('2020-06-09'):
        return 1
    else:
        return 0
def cross_val(param_grid,df_cross_val,stream):
  # Generate all combinations of parameters
  all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
  rmses = []  # Store the RMSEs for each params here

  # Use cross validation to evaluate all parameters
  for params in all_params:
      print(params)
      m = Prophet(**params)  # Fit model with given params
      m.add_country_holidays(country_name='RU')
      if stream == 'partners_mobile':
        df_cross_val['regressor'] = df_cross_val['ds'].apply(add_regressor)
        m.add_regressor('regressor')
      if stream == 'lenta_main_mail_ru':
        df_cross_val['regressor'] = df_cross_val['ds'].apply(add_regressor1)
        m.add_regressor('regressor', prior_scale=10, mode='multiplicative')
      m.fit(df_cross_val)
      df_cv = cross_validation(m, horizon='20 days', parallel="processes")
      df_p = performance_metrics(df_cv, rolling_window=1)
      rmses.append(df_p['mape'].values[0])

  # Find the best parameters
  tuning_results = pd.DataFrame(all_params)
  tuning_results['mape'] = rmses
  print(tuning_results)
  print(tuning_results[np.argmin(rmses):np.argmin(rmses)+1])
  return all_params[np.argmin(rmses)]

def f(x, a, b,c,d):
    return a/(b*x+c)+d

def make_polynom_prediction(df_full):
  df_poly = df_full[-203:]
  max=np.percentile(df_poly['y'].values,95)
  min=np.percentile(df_poly['y'].values,5)
  df_poly = df_poly.loc[(df_poly['y']<max) & (df_poly['y']>min)]

  x = [i+1 for i in range(len(df_poly))]
  y=df_poly.y.values

  popt, _ = curve_fit(f, x, y, method='trf',bounds=([0,-np.inf,-np.inf,0],[np.inf,np.inf,np.inf,np.inf]))

  fit_y = [f(xi, popt[0], popt[1], popt[2], popt[3]) for xi in x]
  prediction = [f(i, popt[0], popt[1], popt[2], popt[3]) for i in range(len(df_poly),len(df_poly)+500,1)]
  #import matplotlib.pyplot as plt
  #plt.plot(x, y, 'o', x, fit_y, '-')
  #plt.show()
  return prediction

def predict_dau_prophet(df_dau,list_of_df,stream):
    param_grid = {
        'yearly_seasonality': [True],
        'weekly_seasonality': [True],
        'changepoint_prior_scale': [0.001, 0.01, 0.5, 10],  # [0.001, 0.5]
        'seasonality_prior_scale': [0.01, 0.1, 1, 10.0],  # [0.01, 10]
        'holidays_prior_scale': [0.01, 0.1, 1, 10]  # [0.01, 10]
    }
    #growth = 'linear'
    if stream == "startsWith(stream_id, 'xiaomi_browser')" or stream == "startsWith(stream_id, 'xiaomi_appvault')" or stream == "startsWith(stream_id, 'xiaomi_lockscreen')":
        #growth = 'logistic'
        best_params = {'yearly_seasonality': False, 'weekly_seasonality': True, 'changepoint_prior_scale': 0.051,
                       'seasonality_prior_scale': 0.01, 'holidays_prior_scale': 1.0}
    elif stream == "startsWith(stream_id, 'lenta_main_mail_ru') and stream_id != 'lenta_main_mail_ru_mediaproject'":
     #   best_params = {'yearly_seasonality': True, 'weekly_seasonality': True, 'changepoint_prior_scale': 0.1,
    #                   'seasonality_prior_scale': 0.02, 'holidays_prior_scale': 0.001}
        best_params={'yearly_seasonality': True, 'weekly_seasonality': True, 'changepoint_prior_scale': 0.01,
                       'seasonality_prior_scale': 3, 'holidays_prior_scale': 0.01}
    #best_params: {'yearly_seasonality': True, 'weekly_seasonality': True, 'changepoint_prior_scale': 0.5,
    #              'seasonality_prior_scale': 0.01, 'holidays_prior_scale': 0.01}

    elif stream == 'partners_mobile':
        best_params = {'yearly_seasonality': True, 'weekly_seasonality': True, 'changepoint_prior_scale': 10,
                       'seasonality_prior_scale': 0.01, 'holidays_prior_scale': 0.01}
    elif stream == 'partners_desktop':
        best_params = {'yearly_seasonality': True, 'weekly_seasonality': True, 'changepoint_prior_scale': 0.5,
                       'seasonality_prior_scale': 1, 'holidays_prior_scale': 10}
    else:
        best_params = cross_val(param_grid, df_dau, stream)

    print('best_params: ', best_params)
    m = Prophet(**best_params)
    m.add_country_holidays(country_name='RU')
    if stream == 'partners_mobile':
        df_dau['regressor'] = df_dau['ds'].apply(add_regressor)
        m.add_regressor('regressor')
    #if stream == "startsWith(stream_id, 'lenta_main_mail_ru') and stream_id != 'lenta_main_mail_ru_mediaproject'":
    #    print('add_regressor1')
    #    df_dau['regressor'] = df_dau['ds'].apply(add_regressor1)
    #    m.add_regressor('regressor', prior_scale=10, mode='multiplicative')

    df_dau = df_dau[df_dau.ds < pd.to_datetime(PREDICT_FROM)]

    #df_dau['cap'] = 3300000
    #df_dau['floor'] = 2700000

    m.fit(df_dau)
    future = m.make_future_dataframe(periods=500, freq='D')
    if stream == 'partners_mobile':
        future['regressor'] = future['ds'].apply(add_regressor)
    #if stream == "startsWith(stream_id, 'lenta_main_mail_ru') and stream_id != 'lenta_main_mail_ru_mediaproject'":
     #   future['regressor'] = future['ds'].apply(add_regressor1)
    #future['cap'] = 3300000
    #future['floor'] = 2700000
    forecast = m.predict(future)
    #fig = m.plot(forecast)
    #a = add_changepoints_to_plot(fig.gca(), m, forecast)

    #fig = m.plot_components(forecast)

    forecast2 = forecast[forecast.ds > pd.to_datetime(df_dau['ds'][-1:].values[0])]
    dau = forecast2.copy()  # save dau for using later with coefs
    #forecast2['date'] = forecast2['ds'].dt.month
    #forecast2['year'] = forecast2['ds'].dt.year
    #forecast2 = forecast2[['date', 'year', 'yhat']].groupby(['year', 'date']).mean()
    forecast2['stream_condition'] = stream
    #forecast2['condition'] = condition
    forecast2['predict_date'] = PREDICT_FROM
    forecast2['yhat'] = forecast2['yhat'].astype('int')
    list_of_df.append(forecast2[['ds', 'predict_date', 'stream_condition', 'yhat']].rename(columns = {'ds': 'date', 'yhat': 'DAU'}).reset_index())
    return dau

def predict_metrics_polynom(df, list_of_df,dau):
    # модель с гиперболой, средним и настройками ленты
    for feature in df.columns:
        if feature in ['DAU', 'date']:
            continue
        df_feature = df.loc[:, ['date', feature]]

        df_feature.columns = ['ds', 'y']
        # sns.lineplot(df_feature.ds, df_feature.y)
        if feature == 'DAU':
            pass
        else:
            df_feature['y'] = df['DAU'] / df_feature['y']
            coef = df_feature['y'][-28:].mean()
            coef7 = df_feature['y'][-7:].mean()
            #print(feature)
            #print('coef', coef)
            #print('coef7', coef7)
            try:
                prediction = make_polynom_prediction(df_feature)
            except:
                prediction = [0]
            error = np.abs((coef - prediction[0]) / coef)
            #print('error', error)
            if error > 0.3:
                #print(feature, 'using 7 days mean:', coef7)
                forecast = dau.copy()
                forecast['yhat'] = forecast['yhat']/coef7
                forecast = forecast.reset_index()
                #forecast['yhat'] = coef7
            else:
                #print(feature, 'using polynom prediction')
                #print('prediction', prediction[0])
                lenta_settings_coef = coef7 / prediction[0]
                #print('lenta_settings_coef', lenta_settings_coef)
                forecast = dau.copy()
                forecast = forecast[forecast.ds > pd.to_datetime(df['date'][-1:].values[0])]
                forecast = forecast.reset_index()
                polynom_prediction_df = pd.DataFrame({feature: prediction})
                # forecast['coef'] = polynom_prediction_df[feature]
                forecast['yhat'] = forecast['yhat']/(polynom_prediction_df[feature] * lenta_settings_coef)
                #forecast['yhat'] = polynom_prediction_df[feature] * lenta_settings_coef
        forecast2 = forecast[forecast.ds > pd.to_datetime(df['date'][-1:].values[0])]
        # if feature == 'DAU': DAU = forecast2.copy() #save dau for using later with coefs
        #forecast2['date'] = forecast2['ds'].dt.month
        #forecast2['year'] = forecast2['ds'].dt.year
        #forecast2 = forecast2[['date', 'year', 'yhat']].groupby(['year', 'date']).mean()
        #forecast2['yhat'].astype('int', copy=False)
        list_of_df.append(forecast2['yhat'].astype('int').rename(feature))

    df_final = pd.concat(list_of_df, axis=1)
    return df_final

def write_df_to_ch(df_final,table_name):
    #df_final.to_excel('/Users/l.pozdnyakov/Desktop/for_pred/pred_'+str(pd.to_datetime(PREDICT_FROM).month-1)+'.xlsx', engine='xlsxwriter')
    df_final = df_final.fillna(0).set_index('date').drop(['index'], axis=1)
    connection = {'host': ' http://proxy.surfy.ru:8125/?user=&password=',
                  'database': 'pulse'}

    affected_rows = ph.to_clickhouse(df_final, table=table_name, connection=connection)

def main():
    print('PREDICT_FROM: ', PREDICT_FROM)
    delete()
    query_id = 8787  # id запроса в redash",
    streams_conditions = [
        "startsWith(stream_id, 'lenta_main_mail_ru') and stream_id != 'lenta_main_mail_ru_mediaproject'",
        "startsWith(stream_id, 'xiaomi_browser')"
    ]
    #conditions = ['1=1']
    df_stream = []
    for stream_condition in streams_conditions:
      #for condition in conditions:
        readsh_params = {
            'p_stream_condition': stream_condition
            #'p_condition': condition
        }
        print(readsh_params)
        query_res = get_redash_result(query_id, readsh_params)
        df = pd.DataFrame(query_res)

        list_of_df = []
        df.date = pd.to_datetime(df.date)#, format='%yyyy-%mm-%dd')
        df = df.sort_values(by='date')
        df = df[df.date < pd.to_datetime(PREDICT_FROM)]

        if stream_condition=="startsWith(stream_id, 'lenta_main_mail_ru') and stream_id != 'lenta_main_mail_ru_mediaproject'":
            df['DAU'] = df['DAU_hit']
        df = df.drop(['DAU_hit'], axis=1)

        df_dau = df.loc[:, ['date', 'DAU']]

        df_dau.columns = ['ds', 'y']
        dau = predict_dau_prophet(df_dau, list_of_df, stream_condition)
        df_stream += [predict_metrics_polynom(df, list_of_df, dau)]
        df_final = pd.concat(df_stream)
    #print(df_final)
    write_df_to_ch(df_final, 'stats_prediction_replicated_distributed')


if __name__ == "__main__":
    main()
