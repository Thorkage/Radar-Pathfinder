from helper_functions import *
from add_dataflashlog import *
from attach_grounddata import *

def load_campaign_config(file_path):
    """
    
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_RADAR(radar_type='', #UWiBaSS
                   datasetID='',
                   campaignID='',
                   path='',
                   yaml_file="./campaigns.yaml"
                   ):
    """
    
    """
    
    config = load_campaign_config(yaml_file)   
    
    if radar_type == 'UWiBaSS':
        full_path = os.path.join(config['master_path'], 'UWiBaSS', datasetID , 'uwibass_object.pkl')
    else:
        full_path = path
        
    with open(full_path, 'rb') as inp:
        RADAR = pickle.load(inp)
        
    RADAR.radar_type = radar_type
    RADAR.datasetID = datasetID
    RADAR.campaignID = campaignID
    RADAR.data_path = config['master_path']
    RADAR.config_paths = config
    RADAR.full_path = full_path
    
    print(f'Dataset {RADAR.datasetID} from campaign {RADAR.campaignID} is loaded from {full_path}')

    return RADAR


def populate_datafields(RADAR,
                        PF_parameters,
                        force_RPCA=False,
                        RPCA_lambda=0.5,
                        add_insitu_data=False,
                        check_dataflashlog=False,
                        has_flightstate=False,
                        has_geolocation=False
                        ):
    """
    
    
    """

    RADAR.has_flightstate = has_flightstate
    RADAR.has_geolocation = has_geolocation
    RADAR.add_insitu_data = add_insitu_data
    RADAR.check_dataflashlog = check_dataflashlog
    
    RADAR = _add_RPCA(RADAR, RPCA_lambda, force_RPCA=force_RPCA)
    
    if RADAR.radar_type == 'UWiBaSS' and has_flightstate and has_geolocation:
        RADAR = _check_auxilliary(RADAR, check_dataflashlog)
    
    RADAR.PF_parameters = PF_parameters

    if has_geolocation:
        RADAR, UTM_transformer, _ = _add_UTM(RADAR)
        RADAR = _add_target_type(RADAR)

        if has_flightstate:
            RADAR = _add_radar_footprints(RADAR)
            
    else:
        RADAR.target_type = 'unknown'
        
    if add_insitu_data and has_geolocation:
        RADAR, df_MP, df_SP, dict_SP = _add_insitu(RADAR, UTM_transformer)
        return RADAR, df_MP, df_SP, dict_SP
    
    else:
        RADAR.SP_bulk_eps_r = np.ones(RADAR.rx_rpca.shape[1]) * 1.5
        RADAR.SP_bulk_eps_r_uncertainty = np.ones(RADAR.rx_rpca.shape[1]) * 0.1
        return RADAR, None, None, None


def _check_auxilliary(RADAR, check_dataflashlog):
    """
    Sanity check of UWiBaSS specific stuff.
    Like the outliers in the laser altimeter data, the constant pitch angle, etc. (add more if you want)
    
    If wanted, the whole dataflashlog association can be repeated here.
    """
    RADAR.datetime_timestamp = [datetime.fromtimestamp(timestamp) for timestamp in RADAR.timestamp.flatten()]
    
    if check_dataflashlog:
        RADAR = _check_dataflashlog(RADAR)

    RADAR.compensated_pitch = RADAR.pitch + 6
    RADAR.CTUN_SAlt_filtered = remove_altitude_outliers(RADAR.CTUN_SAlt, diff_threshold=.25)
    
    if 'target_type' in RADAR.__dict__:
        expected_snow_depth = 2.5 if RADAR.target_type == 'terrestrial' or RADAR.target_type == 'unknown' else 0.5
    else:
        RADAR = _add_target_type(RADAR)
        expected_snow_depth = 2.5 if RADAR.target_type == 'terrestrial' or RADAR.target_type == 'unknown' else 0.5
        
    RADAR.altitude_mask = (RADAR.CTUN_SAlt_filtered < 2 * (RADAR.range_air[-1] / 100) - 2 * expected_snow_depth) & (RADAR.CTUN_SAlt_filtered > 1 * (RADAR.range_air[-1] / 100) + expected_snow_depth)
    return RADAR

def _check_dataflashlog(RADAR):
    """
    
    """
    
    #! To change stuff in the processing of the dataflashlog itself use the notebook "Notebooks/converting_dataflashlogs2csv.ipynb"
    dataflashlog_variables = [
        'GPS.Lat', 'GPS.Lng', 'GPS.Status', 'GPS.Alt',
        'CTUN.Alt', 'CTUN.SAlt', 'POS.Alt', 'BARO.Alt',
        'UTM_x', 'UTM_y', 
    ]
    RADAR, dataflashlog_file = find_dataflashlog(RADAR)
    RADAR = attach_dataflashlog(RADAR, load_dataflashlog(RADAR),
                                dataflashlog_variables)
    return RADAR


def _add_UTM(RADAR):
    """
    Calculate the best UTM projection of the median geographic coordinates and add to the object.
    Additionally, get a epsg:4326 to UTM transformer for later use.
    """
    zone_number = utm.latlon_to_zone_number(np.nanmedian(RADAR.GPS_Lat), np.nanmedian(RADAR.GPS_Lng))
    RADAR.UTM_zone = zone_number
    transformer = Transformer.from_crs(4326, CRS.from_proj4(f"+proj=utm +zone={RADAR.UTM_zone} +ellps=WGS84 +datum=WGS84 +units=m +type=crs"), always_xy=True)
    reverse_transformer = Transformer.from_crs(CRS.from_proj4(f"+proj=utm +zone={RADAR.UTM_zone} +ellps=WGS84 +datum=WGS84 +units=m +type=crs"), 4326, always_xy=True)
    return RADAR, transformer, reverse_transformer

def _add_target_type(RADAR):
    """
    Determine the target type of the dataset based on its geographical location.
    We make use of the Basemap package, this is somewhat deprecated -- but it is the only one I found to be working for the small fjords on Svalbard.
    """
    bm = Basemap(epsg=4326,
        resolution='l', # this is l = low resolution, it's enough for everything I have tested. 
        llcrnrlon = RADAR.GPS_Lng.min() - 1,
        urcrnrlon = RADAR.GPS_Lng.max() + 1,
        llcrnrlat = RADAR.GPS_Lat.min() - 1,
        urcrnrlat = RADAR.GPS_Lat.max() + 1,
        )

    is_seaice = not bm.is_land(np.nanmedian(RADAR.GPS_Lng), np.nanmedian(RADAR.GPS_Lat))
    RADAR.target_type = 'sea_ice' if is_seaice else 'terrestrial_snow'
    return RADAR


def _add_RPCA(RADAR,
              RPCA_lambda,
              force_RPCA=False
              ):
    """
    Add RPCA (Robust Principal Component Analysis) filtered radar-echogram to the RADAR object.
    """
    RADAR.rx_rpca_lambda = RPCA_lambda

    if ('rx_rpca' not in RADAR.__dict__) or force_RPCA == True:
        print('Computing RPCA of radar data...')
        
        if 'rxraw' in RADAR.__dict__:
            _, S = rpca_pcp_ialm(
                    RADAR.rxraw,
                    sparsity_factor= RPCA_lambda / np.sqrt(max(RADAR.rxraw.shape)), 
                    max_iter=100,
                    verbose=False
            )
            RADAR.rx_rpca = S.copy()
        elif 'rx' in RADAR.__dict__:
            _, S = rpca_pcp_ialm(
                    RADAR.rx,
                    sparsity_factor= RPCA_lambda / np.sqrt(max(RADAR.rx.shape)),
                    max_iter=100,
                    verbose=False
            )
            RADAR.rx_rpca = S.copy()
            
        # if 'rx1' in RADAR.__dict__ and 'rx2' in RADAR.__dict__:
            # _, S = rpca_pcp_ialm(
            #         RADAR.rx1,
            #         sparsity_factor= RPCA_lambda / np.sqrt(max(RADAR.rx1.shape)),
            #         max_iter=100,
            #         verbose=True
            # )
            # RADAR.rx_rpca = RADAR.rx1.copy()

            # _, S = rpca_pcp_ialm(
            #         RADAR.rx2,
            #         sparsity_factor= RPCA_lambda / np.sqrt(max(RADAR.rx2.shape)),
            #         max_iter=100,
            #         verbose=False
            # )
            # RADAR.rx2_rpca = S.copy()

        with open(RADAR.full_path, 'wb') as outp:
            pickle.dump(RADAR, outp, pickle.HIGHEST_PROTOCOL)        
    return RADAR
 
 
# TODO: Allow for fixed radius footprint if HPBW is unknown
def _add_radar_footprints(RADAR):
    """
    """
    RADAR.footprints, origins = calculate_radar_footprints(RADAR)
    return RADAR
 
 
def _add_insitu(RADAR, transformer):
    """
    Add in-situ measurements to the RADAR object.
    """
    print('Co-locating in-situ data to radar footprints..')
    df_MP = load_MP_data(RADAR, transformer)
    df_SP, dict_SP = load_SP_data(RADAR, transformer, read_mode='full')
    RADAR, df_SP, dict_SP = colocate_MP_and_SP_to_RADAR(RADAR, transformer, df_MP, df_SP, dict_SP)
    
    return RADAR, df_MP, df_SP, dict_SP




def calculate_SR_precision(RADAR, P_as, P_si, P_noise_air, P_noise_snow, k=1): 
    """ 
    Calculate the precision of the picked layers based on the SNR of the picked layers.
    Equations based on Kingsley & Quegan (1998) and Newman et al (2014).
    """

    range_resolution_air = (k * speed_of_light) / (2 * RADAR.bandwidth)
    range_resolution_snow = (k * speed_of_light) / (2 * RADAR.bandwidth) * 1 / np.sqrt(np.mean(RADAR.SP_bulk_eps_r))
    
    eps_samp = (RADAR.dft * speed_of_light) / 2

    SNR_AS = P_as / P_noise_air
    SNR_SI = P_si / P_noise_snow
    
    eps_SNR_as = range_resolution_air / np.sqrt(2 * SNR_AS)
    eps_SNR_si = range_resolution_snow / np.sqrt(2 * SNR_SI)
    eps_SNR_as = np.where(SNR_AS <= 1, range_resolution_air, eps_SNR_as)
    eps_SNR_si = np.where(SNR_SI <= 1, range_resolution_snow, eps_SNR_si)

    eps_SR = np.sqrt(eps_samp**2 + eps_SNR_as**2 + eps_SNR_si**2)
    
    #TODO How to deal with SNR <= 1? (i.e. where the path is "bridging" between areas of clear signal)
    # solution for now is setting the uncertainty to the snow depth / 2.

    eps_SR = np.where((SNR_AS > 1) | (SNR_SI > 1), eps_SR, RADAR.PF_snow_depth/2)

    return eps_SR, eps_samp, eps_SNR_as, eps_SNR_si, range_resolution_air, range_resolution_snow



def calculate_radar_uncertainties(RADAR, section, unbiased_cost):
    """
    
    """
    P_as = pd.Series(section[RADAR.PF_top_interface.astype(int), np.arange(unbiased_cost.shape[1])]).rolling(window=5, center=True, min_periods=0).mean().values + 1e-9
    P_si = pd.Series(section[RADAR.PF_bottom_interface.astype(int), np.arange(unbiased_cost.shape[1])]).rolling(window=5, center=True, min_periods=0).mean().values + 1e-9

    air_signal = flatten([
        section[:RADAR.PF_top_interface[i], i]
        for i in range(section.shape[1])
    ])

    flattened_section = flatten_to_interface(section, RADAR,
                        interface='top',
                        smoothing_window=3,
                        reduce=False
                        )

    flattened_bottom_trace = pd.Series(RADAR.PF_bottom_interface).rolling(window=3, min_periods=0, center=True).mean().astype(int)- pd.Series(RADAR.PF_top_interface).rolling(window=3, min_periods=0, center=True).mean().astype(int)

    snow_signal = mask_path_in_cost(flattened_section,
                                zip(flattened_bottom_trace, range(len(RADAR.PF_bottom_interface))),
                                radius=2, #TODO: BASE THIS ON PHYSICS (Expected range resolution)
                                strength=np.inf
                                )
    snow_signal = mask_cost_below_above_path(snow_signal,
                                            zip(flattened_bottom_trace, range(len(RADAR.PF_bottom_interface))),
                                            np.inf,
                                            layer='bottom'
                                            )

    snow_signal[np.isinf(snow_signal)] = np.nan
    snow_signal = snow_signal.flatten()
    snow_signal = snow_signal[~np.isnan(snow_signal)]

    global_noise_air = np.nanstd(air_signal)
    global_noise_snow = np.nanstd(snow_signal)

    return calculate_SR_precision(RADAR,
                                P_as=P_as,
                                P_si=P_si,
                                P_noise_air=global_noise_air,
                                P_noise_snow=global_noise_snow
                                )

    
def calculate_snow_depth_uncertainty(RADAR):    
    """
    
    """
    return np.sqrt(
        RADAR.PF_radar_uncertainty**2 / RADAR.SP_bulk_eps_r +\
        RADAR.PF_snow_depth_air**2 / (RADAR.SP_bulk_eps_r**2) * RADAR.SP_bulk_eps_r_uncertainty**2
    )
    

def calculate_total_uncertainty(RADAR, validation_bias=0.01, MP_precision=0.001):
    """
    
    """
    return np.sqrt(
        RADAR.PF_snow_uncertainty ** 2 + validation_bias ** 2 + MP_precision ** 2
    )

# def eps_r_tiuri(bulk_density):
#     return 1 + 1.7 * bulk_density/1000 + 0.7 * (bulk_density/1000)**2

def eps_r_ulaby(bulk_density):
    """
    
    """
    bulk_density = np.array(bulk_density)
    return (1 + 0.51 * bulk_density / 1000) ** 3

def eps_r_ulaby_reversed(eps_r):
    """
    
    """
    return  (eps_r ** (1/3) - 1) / (0.51/1000)

def eps_r_uncertainty_ulaby(bulk_density, bulk_density_unc):
    """
    
    """
    factor = (1 + 0.51 * bulk_density / 1000)
    derivative = 3 * factor**2 * 0.51 / 1000
    return abs(derivative) * bulk_density_unc

def _construct_radar_footprint(
    altitude,
    utm_x,
    utm_y,
    yaw,
    pitch,
    roll,
    E_HPBW=45, # these are the half-power beam-widths (HPBW) in H- and E-plane averaged over the radar bandwidth according to Jenssen et al (2022)
    H_HPBW=25,
    num_rays=100,
    plot=False
    ):
    
    """
    Compute a cone intersecting a plane and estimate that as the radar footprint.
    All attitude angles are respected in the cone's rotation.
    """
    theta_E = np.deg2rad(E_HPBW / 2)
    theta_H = np.deg2rad(H_HPBW / 2)

    #(1): Unit direction vectors to construct the elliptical cone
    angles = np.linspace(0, 2 * np.pi, num_rays)
    x_dir = np.tan(theta_E) * np.cos(angles)
    y_dir = np.tan(theta_H) * np.sin(angles)
    z_dir = -np.ones_like(angles)

    rays = np.stack([x_dir, y_dir, z_dir], axis=1)
    rays /= np.linalg.norm(rays, axis=1)[:, np.newaxis]  # Normalize direction vectors

    # (2): Rotate cone by yaw (z), pitch (y), roll (x)
    rotation = R.from_euler('zyx', [-yaw, pitch, roll], degrees=True)
    rays_rot = rotation.apply(rays)

    # (3): Intersect with ground (z=0): find scale factors
    scales = altitude / (-rays_rot[:, 2])  # stop at z=0
    ground_points = rays_rot * scales[:, np.newaxis]

    # (4): Shift to radar location in UTM
    ground_points[:, 0] += utm_x
    ground_points[:, 1] += utm_y

    # (5): Create polygon from ground points
    footprint_poly = Polygon(ground_points[:, :2])
    
    # Radar origin
    origin = np.array([utm_x, utm_y, altitude])

    return footprint_poly, origin


def calculate_radar_footprints(RADAR):  
    """
    
    """
    footprints = []
    origins = []
    for i in range(RADAR.rx_rpca.shape[1]):
        footprint_poly, origin = _construct_radar_footprint(
                                    altitude=RADAR.CTUN_SAlt_filtered[i],
                                    utm_x=RADAR.UTM_x[i],
                                    utm_y=RADAR.UTM_y[i],
                                    yaw=RADAR.yaw[i],
                                    pitch=RADAR.compensated_pitch[i],
                                    # roll=uwibass.roll[i],
                                    roll=0,  # ! Assuming roll is not used (compensated for by the SnowDrone) --> set to zero
                                    num_rays=200,
                                    plot=False
                                    )
        footprints.append(footprint_poly)
        origins.append(origin)
        
        
    return footprints, origins