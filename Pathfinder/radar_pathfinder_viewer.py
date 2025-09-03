import warnings
warnings.filterwarnings("ignore")

from helper_functions import *
from radar_functions import *
from pathfinder import *
from attach_grounddata import *
from add_dataflashlog import *
from clicki_tool import *


import argparse
import matplotlib
matplotlib.use('qt5agg')

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=str, default='..Data/SnowDrone/UWIBASS/', help="Directory in which RADAR data is located")
parser.add_argument("--datasetID", type=str, default='26_04_2024_5', help="Dataset ID (e.g. 26_04_2024_5)")
parser.add_argument("--campaignID", type=str, default='2024_ESASnowDrone', help="Campaign ID (e.g. 2024_ESASnowDrone)")

args = parser.parse_args()

directory = args.directory 
datasetID = args.datasetID
campaignID = args.campaignID

print("Directory:", directory)
print("DatasetID:", datasetID)
print("CampaignID:", campaignID)

RADAR = load_RADAR(radar_type='UWiBaSS', datasetID=datasetID, campaignID=campaignID)
RADAR, df_MP, df_SP, dict_SP = populate_datafields(RADAR,
                                     PF_parameters=RADAR.PF_parameters,
                                     has_flightstate=RADAR.has_flightstate,
                                     has_geolocation=RADAR.has_geolocation,
                                     add_insitu_data=RADAR.add_insitu_data,
                                     check_dataflashlog=RADAR.check_dataflashlog,
                                     )

clicki_tool = LayerCorrectionTool(RADAR,
                                  has_geolocation=RADAR.has_geolocation,
                                  has_flightstate=RADAR.has_flightstate,
                                  add_grounddata=RADAR.add_insitu_data,
                                  df_MP=df_MP if RADAR.add_insitu_data else None,
                                  df_SP=df_SP if RADAR.add_insitu_data and df_SP is not None else None,
                                  dict_SP=dict_SP if RADAR.add_insitu_data and dict_SP is not None else None
                                  )