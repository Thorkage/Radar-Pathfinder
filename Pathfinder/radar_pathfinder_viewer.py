#FOR WIDGET/ NOTEBOOK PLOTTING
import warnings

from helper_functions import *
del sys.modules['helper_functions']
from helper_functions import *

from pathfinder import *
del sys.modules['pathfinder']
from pathfinder import *

from attach_grounddata import *
del sys.modules['attach_grounddata']
from attach_grounddata import *

from add_dataflashlog import *
del sys.modules['add_dataflashlog']
from add_dataflashlog import *

# from twoD_fourier_filtering import *
# del sys.modules['twoD_fourier_filtering']
# from twoD_fourier_filtering import *

from clicki_tool import *
del sys.modules['clicki_tool']
from clicki_tool import *

# from FKmig import *
# del sys.modules['FKmig']
# from FKmig import *


warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('qt5agg')

import matplotlib as mpl
mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'  



add_grounddata = True
standard_eps_r = 1.5

dir = '/Volumes/PortableSSD/UWIBASS/'
# dir = '/Users/torka/NORCE_T/St3TART-FO/Uwibass'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=str, default='/Volumes/PortableSSD/UWIBASS/', help="Directory in which radar data is located")
parser.add_argument("--dataset", type=str, default='26_04_2024_5', help="Dataset ID (e.g. 26_04_2024_5)")
parser.add_argument("--campaign", type=str, default='ESASnowDrone2024', help="Campaign ID (e.g. ESASnowDrone2024)")

args = parser.parse_args()

directory = args.directory 
dataset = args.dataset
campaign = args.campaign

print("Directory:", directory)
print("Dataset:", dataset)
print("Campaign:", campaign)


full_path = os.path.join(directory,dataset,'uwibass_object.pkl')

with open(full_path, 'rb') as inp:
    uwibass = pickle.load(inp)
    

print(f"Dataset loaded: {os.path.join(dir,dataset,'uwibass_object.pkl')}\nLog file: {uwibass.dataflashlog_path}\n{len(uwibass.timestamp)} columns")
print("-------------------------")

# print(f"UTM zone: {uwibass.utm_zone}")
transformer = Transformer.from_crs(4326, CRS.from_proj4(f"+proj=utm +zone={uwibass.utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +type=crs"), always_xy=True)
reverse_transformer = Transformer.from_crs(CRS.from_proj4(f"+proj=utm +zone={uwibass.utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +type=crs"), 4326, always_xy=True)

# if add_grounddata:

print('Loading and matching in-situ data...')
df_snowprofile, dict_snowprofile = load_snowprofiles(uwibass, campaign, transformer, read_mode='full')
df_MP = load_mp_data(campaign, transformer)
uwibass, df_snowprofile, dict_snowprofile = colocate_MP_and_snowprofiles(uwibass,
                        transformer,
                        df_MP,
                        df_snowprofile,
                        dict_snowprofile,
                        footprint_radius=3.5, #TODO base this on the actual footprint of the radar
                        snowprofile_detail_radius=2, #TODO hmm, a lot of times the snow pit is not very close to the UAV track, needs adjustment depending on the dataset
                        snowprofile_bulk_radius=1500,
                        replacement_eps_r=standard_eps_r,
                        )


tool = LayerCorrectionTool(uwibass,
                           uwibass.data_path,
                           uwibass.rx_rpca,
                           uwibass.PF_top_interface,
                           uwibass.PF_bottom_interface,
                           uwibass.PF_cost_maps['unbiased'],
                           uwibass.PF_cost_maps['top'],
                           uwibass.PF_cost_maps['bottom'],
                           uwibass.PF_internal_layers,
                           window_size=50, 
                           chunk_size=700,
                           overlap=128, 
                           jumpiness=uwibass.PF_parameters['jumpiness'],
                           add_grounddata=add_grounddata,
                           df_MP=df_MP if add_grounddata else None,
                           df_snowprofile=df_snowprofile if add_grounddata and df_snowprofile is not None else None,
                           dict_snowprofile=dict_snowprofile if add_grounddata and dict_snowprofile is not None else None
                           )

