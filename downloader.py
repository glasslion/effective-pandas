import os
import json
import glob
import datetime
from io import StringIO
import concurrent.futures
import zipfile

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_weather(stations, start=pd.Timestamp('2017-01-01'),
                end=pd.Timestamp('2017-01-31')):
    '''
    Fetch weather data from MESONet between ``start`` and ``stop``.
    '''
    url = ("http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
           "&data=tmpf&data=relh&data=sped&data=mslp&data=p01i&data=v"
           "sby&data=gust_mph&data=skyc1&data=skyc2&data=skyc3"
           "&tz=Etc/UTC&format=comma&latlon=no"
           "&{start:year1=%Y&month1=%m&day1=%d}"
           "&{end:year2=%Y&month2=%m&day2=%d}&{stations}")

    stations = "&".join("station=%s" % s for s in stations)
    # print(url.format(start=start, end=end, stations=stations))
    weather = (pd.read_csv(url.format(start=start, end=end, stations=stations),
                           comment="#")
                 .rename(columns={"valid": "date"})
                 .rename(columns=str.strip)
                 .assign(date=lambda df: pd.to_datetime(df['date']))
                 .set_index(["station", "date"])
                 .sort_index())
    float_cols = ['tmpf', 'relh', 'sped', 'mslp', 'p01i', 'vsby', "gust_mph"]
    weather[float_cols] = weather[float_cols].apply(pd.to_numeric, errors="corce")
    return weather


def get_weather_ids(network):
    url = "http://mesonet.agron.iastate.edu/geojson/network.php?network={}"
    r = requests.get(url.format(network))
    md = pd.io.json.json_normalize(r.json()['features'])
    md['network'] = network
    return md

def weather_worker(k, v):
    weather = get_weather(v['id'])
    weather.to_csv("data/weather/{}.csv".format(k))


def download_weather():
    states = """AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME
 MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT
 WA WI WV WY""".split()

    # IEM has Iowa AWOS sites in its own labeled network
    networks = ['AWOS'] + ['{}_ASOS'.format(state) for state in states]

    ids = pd.concat(
        [get_weather_ids(network) for network in networks], ignore_index=True)
    gr = ids.groupby('network')

    store = 'data/weather.h5'
    if not os.path.exists(store):
        os.makedirs("data/weather", exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as e:
            fs = [e.submit(weather_worker, k, v) for k, v in gr]
            with tqdm(total=len(fs), desc="Downloading") as pbar:
                for i, future in enumerate(concurrent.futures.as_completed(fs)):
                    pbar.update(i)

        weather = pd.concat([
            pd.read_csv(f, parse_dates=['date'], index_col=['station', 'date'])
            for f in glob.glob('data/weather/*.csv')
        ]).sort_index()

        weather.to_hdf("data/weather.h5", "weather")
    else:
        weather = pd.read_hdf("data/weather.h5", "weather")
    return weather


def download_flights():
    headers = {
    'Referer': 'https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time',
    'Origin': 'https://www.transtats.bts.gov',
    'Content-Type': 'application/x-www-form-urlencoded',
    }

    params = (
        ('Table_ID', '236'),
        ('Has_Group', '3'),
        ('Is_Zipped', '0'),
    )

    with open('modern-1-url.txt', encoding='utf-8') as f:
        data = f.read().strip()

    os.makedirs('data', exist_ok=True)
    dest = "data/flights.csv.zip"

    if not os.path.exists(dest):
        r = requests.post('https://www.transtats.bts.gov/DownLoad_Table.asp',
                        headers=headers, params=params, data=data, stream=True)

        with open("data/flights.csv.zip", 'wb') as f:
            for chunk in r.iter_content(chunk_size=102400):
                if chunk:
                    f.write(chunk)
    zf = zipfile.ZipFile("data/flights.csv.zip")
    fp = zf.extract(zf.filelist[0].filename, path='data/')
    df = pd.read_csv(fp, parse_dates=["FL_DATE"]).rename(columns=str.lower)
    return df


def download_all():
    download_weather()