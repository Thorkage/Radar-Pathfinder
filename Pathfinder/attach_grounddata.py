from helper_functions import *

def _MP_loader(path, campaignID):
    """
    Magnaprobe file loader which can load from different filetypes:
    - .dat: native MP format
    - .nc:  data delivery format used e.g. by SIOS)
    - .mat: older format, similar to .nc
    
    Parameters
    -----------
    path: full path to the file
    campaignID: a unique campaign identifier (e.g. 2019_SIOS) as in the campaigns.yaml config file
    
    Returns
    -------
    insitu_lat: latitude of the data points
    insitu_lon: longitude of the data points
    insitu_depth: the MP measurement itself (snow depth)
    """
    
    filetype = path.split('.')[-1]

    if filetype == 'dat':
        data = pd.read_table(path, sep=',', header=1, skiprows=(2,3))
        if data.LatitudeDDDDD[0]<1:
            insitu_depth = np.copy(data.DepthCm) #+80
            insitu_lat = np.copy(data.latitude_a + data.LatitudeDDDDD)
            insitu_lon = np.copy(data.Longitude_a + data.LongitudeDDDDD)
        else:
            insitu_depth = np.copy(data.DepthCm) #+80
            insitu_lat = np.copy(data.LatitudeDDDDD)
            insitu_lon = np.copy(data.LongitudeDDDDD)
            
        # The 2019 SIOS campaign has some unique corrections (simply copied from RODJ): 
        if campaignID == '2019_SIOS':
            # datapoints which used the 80 cm extension:
            add80sections = np.array( [(464, 561), (972, 1176)], dtype = int)
            for i in add80sections: 
                insitu_depth[i[0]:i[1]] = insitu_depth[i[0]:i[1]] + 80
            # Delete obvious outliers:
            insitu_remove = np.arange(259,262)
            for index in sorted(insitu_remove, reverse=True):
                insitu_depth = np.delete(insitu_depth, index)
                insitu_lat = np.delete(insitu_lat, index)
                insitu_lon = np.delete(insitu_lon, index)
            # # Delete obvious outliers:
            insitu_lat = np.delete(insitu_lat, np.argwhere(insitu_depth < 0.1))
            insitu_lon = np.delete(insitu_lon, np.argwhere(insitu_depth < 0.1))
            insitu_depth = np.delete(insitu_depth, np.argwhere(insitu_depth < 0.1))
        return insitu_lat, insitu_lon,  insitu_depth

    if filetype == 'nc':
        data = xr.open_dataset(path)
        insitu_lat = data['latitude'].values
        insitu_lon = data['longitude'].values
        insitu_depth = data['depth'].values
        return insitu_lat, insitu_lon,  insitu_depth

    if filetype == 'mat':
        mat = scipy.io.loadmat(path)
        site = path.split('/')[-1].split('.')[0]
        probe = mat[site][2:-1]
        insitu_lat = np.copy(probe[:,1])
        insitu_lon = np.copy(probe[:,2])
        insitu_depth = np.copy(probe[:,0])
        return insitu_lat, insitu_lon,  insitu_depth


def load_MP_data(RADAR, transformer):
    """
    
    """

    for i, path in enumerate(RADAR.config_paths['campaigns'][RADAR.campaignID]['magnaprobe_paths']):
        if i == 0:
            insitu_lat, insitu_lon, insitu_depth = _MP_loader(os.path.join(RADAR.data_path, path), RADAR.campaignID)
        else:
            lat, lon, depth = _MP_loader(os.path.join(RADAR.data_path, path), RADAR.campaignID)
            insitu_lat = np.append(insitu_lat, lat)
            insitu_lon = np.append(insitu_lon, lon)
            insitu_depth = np.append(insitu_depth, depth)
                
    utm_x, utm_y = transformer.transform(insitu_lon, insitu_lat)
    df_MP = pd.DataFrame({
        'lat': insitu_lat,
        'lon': insitu_lon,
        'UTM_x': utm_x,
        'UTM_y': utm_y,
        'snow_depth': insitu_depth
        })        
    return df_MP        



def _SP_meta_loader(path, files, transformer):
    """
    Load all snow profiles as a pandas DataFrame from a given campaign (path and list of files).
    Densities are averaged and we estimate a bulk density and associated refractive index for each snow profile via a Monte Carlo simulation.
    """
    
    df_SP = pd.DataFrame({
        'filename': files,
        'lat': [None] * len(files),
        'lon': [None] * len(files),
        'UTM_x': [None] * len(files),
        'UTM_y': [None] * len(files),
        'time': [None] * len(files),
        'bulk_density': [None] * len(files),
        'bulk_density_unc' : [None] * len(files),
        'eps_r': [None] * len(files),
        'eps_r_unc': [None] * len(files)
        
    })

    for f in files:
        with open(os.path.join(path, f)) as json_file:
            data = json.load(json_file)

        lat = data['position']['latitude']
        lon = data['position']['longitude']
        time = data['profiles'][0]['date']
        
        if 'density' in data['profiles'][0].keys():
            thicknesses = []
            densities = []

            for i in range(len(data['profiles'][0]['density']['elements'][0]['layers'])):
                thicknesses.append(data['profiles'][0]['density']['elements'][0]['layers'][i]['top'] - data['profiles'][0]['density']['elements'][0]['layers'][i]['bottom'])
                densities.append(data['profiles'][0]['density']['elements'][0]['layers'][i]['value'])
                
            densities = np.array(densities)
            thicknesses = np.array(thicknesses)
        
            bulk_density, bulk_density_unc, eps_r, eps_r_unc = monte_carlo_eps_r(thicknesses, densities,  thickness_rel_sigma=0.3, density_rel_sigma=0.08, N=20000)

            df_SP.loc[df_SP['filename'] == f, 'bulk_density'] = bulk_density
            df_SP.loc[df_SP['filename'] == f, 'bulk_density_unc'] = bulk_density_unc
            df_SP.loc[df_SP['filename'] == f, 'eps_r'] = eps_r
            df_SP.loc[df_SP['filename'] == f, 'eps_r_unc'] = eps_r_unc

            df_SP.loc[df_SP['filename'] == f, 'UTM_x'], df_SP.loc[df_SP['filename'] == f, 'UTM_y'] = transformer.transform(lon, lat)
            df_SP.loc[df_SP['filename'] == f, 'lat'] = lat
            df_SP.loc[df_SP['filename'] == f, 'lon'] = lon
            df_SP.loc[df_SP['filename'] == f, 'time'] = pd.to_datetime(time)

    return df_SP.dropna().reset_index(drop=True)



    
def _SP_full_loader(RADAR, path, files, transformer):
    """
    
    """
    dict_SP = {}
    
    for f in files:
        with open(os.path.join(path, f)) as json_file:
            data = json.load(json_file)
            
        data['position']['UTM_x'], data['position']['UTM_y'] = transformer.transform(data['position']['longitude'], data['position']['latitude'])
        
        if 'density' in data['profiles'][0].keys():
        
            # TODO this assumes that the SAME layers are present in density and hardness
            # for ESASnowDrone2024 this is not the case
            
            var_density = data['profiles'][0]['density']
            layers_density = var_density['elements'][0]['layers']
            bottoms_density = [layer['bottom'] for layer in layers_density]
            tops_density = [layer['top'] for layer in layers_density]
            vals_density = [layer['value'] if 'value' in layer.keys() else np.nan for layer in layers_density]
            
            var_hardness = data['profiles'][0]['thickness']
            layers_hardness = var_hardness['elements'][0]['layers']
            bottoms_hardness = [layer['bottom'] for layer in layers_hardness]
            tops_hardness = [layer['top'] for layer in layers_hardness]

            bottoms = np.unique(list(set(bottoms_density + bottoms_hardness)))
            bottoms.sort()
            
            tops = np.unique(list(set(tops_density + tops_hardness)))
            tops.sort()
            
            #? this should handle that case
            # if len(bottoms) > len(bottoms_density) + 1:
            #     bottoms = bottoms_density
            #     tops = tops_density

            layers = []
            for i in range(len(bottoms)):
                layers.append({'bottom': bottoms[i], 'top': tops[i], 'density': np.nan, 'hardness': np.nan})
                
            for layer in layers_density:
                for i in range(len(layers)):
                    if layers[i]['bottom'] == layer['bottom'] and layers[i]['top'] == layer['top']:
                        layers[i]['density'] = layer['value']
                        
            for layer in layers_hardness:
                for i in range(len(layers)):
                    if layers[i]['bottom'] == layer['bottom'] and layers[i]['top'] == layer['top']:
                        layers[i]['hardness'] = layer['value'] if 'value' in layer.keys() else np.nan
                        
            density_values = [layer['density'] if not np.isnan(layer['density']) else np.nanmedian(vals_density) for layer in layers]

            eps_r_values = np.array(list(eps_r_ulaby(density_values)))
            eps_r_values_reversed = np.array(list(reversed(eps_r_ulaby(density_values))))

            
            dzs_reversed = RADAR.range_air[1] / np.sqrt(eps_r_values)

            data['profiles'][0]['eps_r'] = {'type': 'Ulaby1986',
                                                        'elements': [{'meta': {'pointprofile': False}, 'layers': [{'value': val, 'top': top, 'bottom': bot} for val, top, bot in zip(eps_r_values, tops, bottoms)]}]
                                                        }
            data['profiles'][0]['refractive_index'] = {'type': 'Ulaby1986',
                                                        'elements': [{'meta': {'pointprofile': False}, 'layers': [{'value': val, 'top': top, 'bottom': bot} for val, top, bot in zip(np.sqrt(eps_r_values), tops, bottoms)]}]
                                                        }
            
            SP_layers = np.insert(tops, 0, bottoms[0]) 
            SP_layers = list(reversed(np.abs(SP_layers - np.max(SP_layers))))
            SP_thickness = np.diff(SP_layers)

            new_layers = [0]
            for i in range(0, len(SP_thickness)):
                new_layers.append(np.round(SP_thickness[i] / dzs_reversed[i]).astype(int) + new_layers[i])


            data['profiles'][0]['RADAR_range_bins'] = {'type': 'flipped_system',
                                                        'elements': [{'meta': {'pointprofile': False}, 'layers': [{'value': val, 'top': top, 'bottom': bot} for val, top, bot in zip(eps_r_values_reversed, new_layers[:-1], new_layers[1:])]}]
                                                        }
                        
                        
        dict_SP[f] = data
        
    return dict_SP


def load_SP_data(RADAR, transformer, read_mode='meta'):
    """
    
    """
    files = os.listdir(os.path.join(RADAR.data_path, RADAR.config_paths['campaigns'][RADAR.campaignID]['snowprofile_path']))
    files = [f for f in files if f.endswith('.json') and not f.startswith('.')]
    df_SP = _SP_meta_loader(os.path.join(RADAR.data_path, RADAR.config_paths['campaigns'][RADAR.campaignID]['snowprofile_path']), files, transformer)
    
    if read_mode == 'full':
        dict_SP = _SP_full_loader(RADAR, os.path.join(RADAR.data_path, RADAR.config_paths['campaigns'][RADAR.campaignID]['snowprofile_path']), files, transformer)
        return df_SP, dict_SP

    elif read_mode == 'meta':
        return df_SP
    
    else:
        raise ValueError("read_mode must be either 'meta' or 'full'.")
    
    


def colocate_MP_and_SP_to_RADAR(RADAR,
                                transformer,
                                df_MP,
                                df_SP,
                                dict_SP,
                                replacement_eps_r=1.5,
                                replacement_eps_r_unc=0.1,
                                SP_max_distance=2000
                                ):
    """
    
    """
    #(1): Get the correct snow profile for the survey and associate the bulk properties
    if df_SP is not None and not df_SP.empty:
        tree = cKDTree(df_SP[['UTM_x', 'UTM_y']].values)
        all_indices = tree.query_ball_point(np.column_stack([RADAR.UTM_x, RADAR.UTM_y]), r=SP_max_distance)
        mode_value, _ = mode([item for sublist in all_indices for item in sublist])
    
        if np.isnan(mode_value):
            print(f"No snow profiles found within the specified radius, falling back to replacement eps_r value: {replacement_eps_r}")
            RADAR.SP_closest_name = None
            all_eps_r = [replacement_eps_r] * len(RADAR.timestamp)   
            all_eps_r_uncertainty = [replacement_eps_r_unc] * len(RADAR.timestamp)
            df_SP = None
            dict_SP = None
            
        else:
            RADAR.SP_closest_name = df_SP.iloc[mode_value]['filename']
                
            df_SP = df_SP.iloc[mode_value]
            dict_SP = dict_SP[RADAR.SP_closest_name]

            all_eps_r = [df_SP['eps_r']] * len(RADAR.timestamp)
            all_eps_r_uncertainty = [df_SP['eps_r_unc']] * len(RADAR.timestamp)

            # (2): Detailed for snow profile to radar internal comparison
            #      Currently only the layer depths themselves are kept.
            #TODO: Add snow profile parameters (to show in clicki tool)
            
            tree = cKDTree(np.column_stack([RADAR.UTM_x, RADAR.UTM_y]))
            profile_point = np.array([[dict_SP['position']['UTM_x'], dict_SP['position']['UTM_y']]])
            all_indices = tree.query_ball_point(profile_point, r=10)[0]

            SP_top_layers = [i['top'] for i in (dict_SP['profiles'][0]['RADAR_range_bins']['elements'][0]['layers'])]
            SP_bottom_layers = [i['bottom'] for i in (dict_SP['profiles'][0]['RADAR_range_bins']['elements'][0]['layers'])]

            RADAR.SP_indices = all_indices
            RADAR.SP_log_UTM_x = RADAR.UTM_x[all_indices]
            RADAR.SP_log_UTM_y = RADAR.UTM_y[all_indices]
            
            RADAR.SP_top_layers = SP_top_layers
            RADAR.SP_bottom_layers = SP_bottom_layers

            # print(f"Snow profile from {df_SP['name']}")
            
    else:
        all_eps_r = [replacement_eps_r] * len(RADAR.timestamp)
        all_eps_r_uncertainty = [replacement_eps_r_unc] * len(RADAR.timestamp)
    
    
    RADAR.SP_bulk_eps_r = np.array(all_eps_r)
    RADAR.SP_bulk_eps_r_uncertainty = np.array(all_eps_r_uncertainty)


    #(2): Co-locate MP data to radar footprints:
    gdf_poly = gpd.GeoDataFrame({'geometry': RADAR.footprints},
                                geometry='geometry'
                                )
    
    gdf_poly = gdf_poly.reset_index().rename(columns={'index': 'poly_id'})  # <-- explicit ID

    gdf_points = gpd.GeoDataFrame(
        df_MP.copy(),
        geometry=gpd.points_from_xy(df_MP['UTM_x'], df_MP['UTM_y']),
        crs=gdf_poly.crs
        )
    
    joined = gpd.sjoin(gdf_points, gdf_poly.reset_index(drop=False), how='inner', predicate='within')
    
    stats = joined.groupby('poly_id').agg(
        n_MP_points=('snow_depth', 'size'),
        mean_snow_depth=('snow_depth', 'mean')
    )
    stats_full = stats.reindex(gdf_poly.index)
    stats_full['n_MP_points'] = stats_full['n_MP_points'].fillna(0)
    
    RADAR.MP_snow_depth = stats_full['mean_snow_depth'].values
    RADAR.MP_N_points = stats_full['n_MP_points'].values
    
    
    dzs = RADAR.range_air[1] / np.sqrt(RADAR.SP_bulk_eps_r) / 100   
    RADAR.MP_snow_depth_in_range_bins = np.round(RADAR.MP_snow_depth / dzs).astype(int)
    
    return RADAR, df_SP, dict_SP
    
    
    


# def colocate_MP_and_snowprofiles(uwibass,
#                                  transformer,
#                                  df_MP,
#                                  df_snowprofile,
#                                  dict_snowprofile,
#                                  footprint_radius=3.0,
#                                  snowprofile_detail_radius=5.0,
#                                  snowprofile_bulk_radius=2000,
#                                  replacement_eps_r=1.5,
#                                  replacement_eps_r_unc=0.1,
#                                  ):
        
#     #! we just use the log values always (we do so anyway later on)
#     # if use_interp_coords:
#     UTM_x, UTM_y = uwibass.log_UTM_x, uwibass.log_UTM_y
#     # else:
#     #     try:
#     #         UTM_x, UTM_y = uwibass.UTM_x, uwibass.UTM_y
#     #     except:
#     #         UTM_, UTM_y = transformer.transform(uwibass.lon, uwibass.lat)
            
#     # FIRST COLOCATE MAGNAPROBE DATA
#     tree = cKDTree(df_MP[['UTM_x', 'UTM_y']].values)
#     all_indices = tree.query_ball_point(np.column_stack([UTM_x, UTM_y]), r=footprint_radius)

#     interpolated_depths = []
#     # interpolated_eps_r = []
#     min_distances = []
#     neighbor_counts = []
#     all_dists = [[]] * len(UTM_x)  # Initialize a list to hold distances for each UWI
#     i = -1
#     for idx_list, uwi_x, uwi_y in zip(all_indices, UTM_x, UTM_y):
#         neighbor_counts.append(len(idx_list))
#         i+=1

#         if not idx_list:
#             interpolated_depths.append(np.nan)
#             min_distances.append(np.nan)
#             # all_dists[i] = [.nan]
#             continue

#         neighbors = df_MP.iloc[idx_list]
#         dists = np.sqrt((neighbors['UTM_x'] - uwi_x)**2 + (neighbors['UTM_y'] - uwi_y)**2)
#         dists[dists == 0] = 1e-6  # avoid divide-by-zero
#         all_dists[i] = dists.tolist()  # Store distances for this UWI
#         weights = 1 / dists**2
#         weighted_depth = np.sum(weights * neighbors['snow_depth']) / np.sum(weights)
#         # weighted_eps_r = np.sum(weights * neighbors['snowprofile_eps_r']) / np.sum(weights)

#         interpolated_depths.append(weighted_depth)
#         # interpolated_eps_r.append(weighted_eps_r)
#         # 
#         min_distances.append(dists.min())
        

#     uwibass.MP_snow_depth = np.array(interpolated_depths)
#     # uwibass.MP_distance = np.array(min_distances)
#     uwibass.MP_N_neighbors = np.array(neighbor_counts)
#     uwibass.MP_indices = all_indices
#     uwibass.MP_distances = all_dists
    
#     #SECOND COLOCATE SNOWPROFILES
#     if df_snowprofile is not None and not df_snowprofile.empty:
        
#         # bulk for eps_r
#         tree = cKDTree(df_snowprofile[['UTM_x', 'UTM_y']].values)
#         all_indices = tree.query_ball_point(np.column_stack([UTM_x, UTM_y]), r=snowprofile_bulk_radius)

#         mode_value, _ = mode([item for sublist in all_indices for item in sublist])

#         if np.isnan(mode_value):
#             print(f"No snow profiles found within the specified radius, falling back to replacement eps_r value: {replacement_eps_r}")
#             uwibass.closest_snowprofile = None
#             all_eps_r = [replacement_eps_r] * len(uwibass.timestamp)   
#             all_eps_r_uncertainty = [replacement_eps_r_unc] * len(uwibass.timestamp)
#             df_snowprofile = None
#             dict_snowprofile = None
            
#         else:
#             uwibass.closest_snowprofile = df_snowprofile.iloc[mode_value]['filename']
                
#             df_snowprofile = df_snowprofile.iloc[mode_value]
#             dict_snowprofile = dict_snowprofile[uwibass.closest_snowprofile]
            
#             all_eps_r = [df_snowprofile['eps_r']] * len(uwibass.timestamp)
#             all_eps_r_uncertainty = [df_snowprofile['eps_r_unc']] * len(uwibass.timestamp)
            
#             # detailed for snow profile to radar comparison
#             UTM_x, UTM_y = uwibass.log_UTM_x, uwibass.log_UTM_y
#             tree = cKDTree(np.column_stack([UTM_x, UTM_y]))
#             profile_point = np.array([[dict_snowprofile['position']['UTM_x'], dict_snowprofile['position']['UTM_y']]])
#             all_indices = tree.query_ball_point(profile_point, r=snowprofile_detail_radius)[0]

#             SP_top_layers = [i['top'] for i in (dict_snowprofile['profiles'][0]['uwibass_range_bins']['elements'][0]['layers'])]
#             SP_bottom_layers = [i['bottom'] for i in (dict_snowprofile['profiles'][0]['uwibass_range_bins']['elements'][0]['layers'])]

#             uwibass.SP_indices = all_indices
#             uwibass.SP_log_UTM_x = uwibass.log_UTM_x[all_indices]
#             uwibass.SP_log_UTM_y = uwibass.log_UTM_y[all_indices]
            
#             uwibass.SP_top_layers = SP_top_layers
#             uwibass.SP_bottom_layers = SP_bottom_layers
            
#             #TODO these min_distances are from the MP, since I deleted the IDW for the snow profiles.
#             #TODO Could just remove, or find a workaround
#             print(f"Snow profile from {dict_snowprofile['name']}")

#     else:
#         all_eps_r = [replacement_eps_r] * len(uwibass.timestamp)
#         all_eps_r_uncertainty = [replacement_eps_r_unc] * len(uwibass.timestamp)

#     uwibass.PF_snow_profile_eps_r = np.array(all_eps_r)
#     uwibass.PF_snow_profile_eps_r_uncertainty = np.array(all_eps_r_uncertainty)
    
#     dzs = uwibass.range_air[1] / np.sqrt(uwibass.PF_snow_profile_eps_r)        
#     uwibass.MP_snow_depth_in_indices = np.round(uwibass.MP_snow_depth / dzs).astype(int)
    
#     return uwibass, df_snowprofile, dict_snowprofile



def eps_r_ulaby(bulk_density):
    bulk_density = np.array(bulk_density)
    return (1 + 0.51 * bulk_density / 1000) ** 3

def monte_carlo_eps_r(
    thicknesses,
    densities,
    thickness_rel_sigma=0.3,   # 30% as 1σ, TODO base this on the Pathfinder interface variance/ std.dev
    density_rel_sigma=0.08,    # 8% as 1σ (from RMSE ≈ σ if bias≈0), reported by Proksch et al (2016) for the box-cutter and "swiss snow"
    N=20000,
    seed=None
    ):
    """
    
    """
    di = np.asarray(thicknesses, dtype=float)
    rho = np.asarray(densities, dtype=float)
    assert di.shape == rho.shape, "thicknesses and densities must have same shape"
    L = di.size
    rng = np.random.default_rng(seed)

    # helper: lognormal params from coefficient of variation (rel σ)
    def _ln_params_from_cv(x, cv):
        sigma = np.sqrt(np.log1p(cv**2))
        mu = np.log(x) - 0.5 * sigma**2
        return mu, sigma

    # sample thickness (lognormal, independent per layer)
    mu_t, sig_t = _ln_params_from_cv(di, thickness_rel_sigma)
    di_samples = rng.lognormal(mean=mu_t, sigma=sig_t, size=(N, L))

    # sample densities (lognormal, independent per layer)
    mu_r, sig_r = _ln_params_from_cv(rho, density_rel_sigma)
    rho_samples = rng.lognormal(mean=mu_r, sigma=sig_r, size=(N, L))

    # bulk density and permittivity
    w = di_samples
    rho_bulk_samples = np.sum(w * rho_samples, axis=1) / np.sum(w, axis=1)
    eps_r_samples = eps_r_ulaby(rho_bulk_samples)

    return (float(np.mean(rho_bulk_samples)),
            float(np.std(rho_bulk_samples, ddof=1)),
            float(np.mean(eps_r_samples)),
            float(np.std(eps_r_samples, ddof=1)))



# def monte_carlo_eps_r(thicknesses, densities, pertubation=0.3, N=20000):
#     di = thicknesses
#     rho_i = densities
#     di_std = pertubation * di
#     di_samples = np.random.normal(loc=di, scale=di_std, size=(N, di.size))
#     rho_bulk_samples = np.sum(di_samples * rho_i, axis=1) / np.sum(di_samples, axis=1)
#     eta_s_samples = eps_r_ulaby(rho_bulk_samples)

#     return np.mean(rho_bulk_samples), np.std(rho_bulk_samples), np.mean(eta_s_samples), np.std(eta_s_samples)


# def read_snowprofiles_meta(path, files, transformer):
#     """
    
    
#     """
#     df = pd.DataFrame({
#             'filename': files,
#             'lat': [None] * len(files),
#             'lon': [None] * len(files),
#             'UTM_x': [None] * len(files),
#             'UTM_y': [None] * len(files),
#             'time': [None] * len(files),
#             'bulk_density': [None] * len(files),
#             'bulk_density_unc' : [None] * len(files),
#             'eps_r': [None] * len(files),
#             'eps_r_unc': [None] * len(files)
            
#         })

#     for f in files:
#         with open(os.path.join(path, f)) as json_file:
#             data = json.load(json_file)

#         lat = data['position']['latitude']
#         lon = data['position']['longitude']
#         time = data['profiles'][0]['date']
        
#         if 'density' in data['profiles'][0].keys():
#             thicknesses = []
#             densities = []

#             for i in range(len(data['profiles'][0]['density']['elements'][0]['layers'])):
#                 thicknesses.append(data['profiles'][0]['density']['elements'][0]['layers'][i]['top'] - data['profiles'][0]['density']['elements'][0]['layers'][i]['bottom'])
#                 densities.append(data['profiles'][0]['density']['elements'][0]['layers'][i]['value'])
                
#             densities = np.array(densities)
#             thicknesses = np.array(thicknesses)
        
#             bulk_density, bulk_density_unc, eps_r, eps_r_unc = monte_carlo_eps_r(thicknesses, densities,  thickness_rel_sigma=0.3, density_rel_sigma=0.08, N=20000)
            
#             df.loc[df['filename'] == f, 'bulk_density'] = bulk_density
#             df.loc[df['filename'] == f, 'bulk_density_unc'] = bulk_density_unc
#             df.loc[df['filename'] == f, 'eps_r'] = eps_r
#             df.loc[df['filename'] == f, 'eps_r_unc'] = eps_r_unc

#             df.loc[df['filename'] == f, 'UTM_x'], df.loc[df['filename'] == f, 'UTM_y'] = transformer.transform(lon, lat)
#             df.loc[df['filename'] == f, 'lat'] = lat
#             df.loc[df['filename'] == f, 'lon'] = lon
#             df.loc[df['filename'] == f, 'time'] = pd.to_datetime(time)
                
#     return df.dropna().reset_index(drop=True)


def plot_MP_validation(uwibass, ax, mode='snow_depth_best'):

    """

    """
    ax.set_aspect('equal')
    ax.grid(ls=':', axis='y')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    
    if mode == 'interfaces':
        xvar = uwibass.MP_snow_depth_in_indices
        yvar = uwibass.bottom_interface_PF - uwibass.top_interface_PF
        
    elif mode == 'snow_depth':
        xvar = uwibass.MP_snow_depth / 100
        yvar = uwibass.PF_snow_depth_T
    
    elif mode == 'snow_depth_best':
        unique_MP_indices = np.unique([item for sublist in uwibass.MP_indices[uwibass.altitude_mask] for item in sublist])
        internals = uwibass.PF_internal_layers['paths']

        MP_depths = []
        UWB_depths = []
        UWB_uncertainties = []
        colors = []
        maski = []
        j = 0
        
        for ind in unique_MP_indices:
            colors.append('dimgrey')
            maski.append(True)
            
            indices = [i for i, item in enumerate(uwibass.MP_indices) if ind in item]
            
            ind_UWB_depths = np.array([uwibass.PF_snow_depth[i] for i, item in enumerate(uwibass.MP_indices) if ind in item])
            ind_UWB_uncs = np.array([uwibass.PF_radar_uncertainty[i] for i, item in enumerate(uwibass.MP_indices) if ind in item])
                        
            ind_MP_depth = np.nanmedian([uwibass.MP_snow_depth[i] for i, item in enumerate(uwibass.MP_indices) if ind in item])
            ind_MP_index = np.nanmedian([uwibass.MP_snow_depth_in_indices[i] for i, item in enumerate(uwibass.MP_indices) if ind in item])

            if np.all(ind_MP_depth/100 < ind_UWB_depths - uwibass.range_resolution_snow/2):

                #TODO check whether there is an internal layer in the
                # proximity of the MP depth (under-penetration)
                
                
                for path in internals:
                    py, px = list(zip(*path))
                    py = np.array(py)
                    py_extended = np.unique(np.concatenate([py, py-2, py - 1, py + 1, py + 2]))
                    px = np.array(px)
                    
                    if any(i in px for i in indices):
                        
                        if ind_MP_index in py_extended:
                            colors[j] = 'red'
                            maski[j] = False
  
            
            UWB_depths.append(ind_UWB_depths[np.argmin(np.abs(ind_UWB_depths - ind_MP_depth/100))])
            MP_depths.append(ind_MP_depth/100)
            UWB_uncertainties.append(ind_UWB_uncs[np.argmin(np.abs(ind_UWB_depths - ind_MP_depth/100))])
            
            j += 1

        xvar = np.array(MP_depths)
        yvar = np.array(UWB_depths)
        yerr = np.array(UWB_uncertainties)
        maski = np.array(maski)


    else:
        raise ValueError("Mode must be either 'interfaces' or 'snow_depth'.")

    # sns.regplot(x=xvar, y=yvar, scatter_kws={'s': 4, 'color': 'black'}, line_kws={'color': 'red'}, ax=ax)
    # # ax.scatter(uwibass.MP_snow_depth/100, uwibass.snow_depth_T, s=4)

    # xlims = (np.min([np.nanmin(xvar), np.nanmin(yvar[~np.isnan(xvar)])]) -.1, np.max([np.nanmax(xvar), np.nanmax(yvar[~np.isnan(xvar)])]) + .1)
    # ax.set_xlim(xlims)
    # ax.set_ylim(xlims)

    # mask = ~np.isnan(xvar)
    # linr = linregress(xvar[mask], yvar[mask])
    # ax.plot(xlims, linr.intercept + linr.slope * np.array(xlims), color='red')
    # ax.plot([xlims[0], xlims[1]], [xlims[0], xlims[1]], color='grey', linestyle='--')

    # ax.set_xlabel('Magnaprobe snow depth [m]')
    # ax.set_ylabel('UWiBaSS snow depth [m]')
    # RMSE = np.sqrt(np.mean((xvar[mask] - yvar[mask])**2))
    # ax.text(0.05, 0.95, f"N = {len(xvar[mask])}\nR² = {linr.rvalue**2:.2f}\nRMSE = {RMSE:.2f} m", transform=ax.transAxes, va='top', ha='left', size=12)
    
    from scipy import odr

    # sns.regplot(x=xvar, y=yvar,
    #             scatter_kws={'s': 4, 'color': 'grey', 'alpha': .5},
    #             line_kws={'color': 'skyblue', 'alpha': .5, 'label': 'OLS'},
    #             ax=ax)
    
    ax.errorbar(xvar, yvar, yerr=yerr, color='grey', alpha=.5, fmt='none', ls='', lw=1)
    ax.scatter(xvar, yvar, s=20, color=colors, alpha=.9, zorder=1000)
    
    xlims = (np.min([np.nanmin(xvar), np.nanmin(yvar[~np.isnan(xvar)])]) - .1,
            np.max([np.nanmax(xvar), np.nanmax(yvar[~np.isnan(xvar)])]) + .1)
    ax.set_xlim(xlims)
    ax.set_ylim(xlims)

    mask = ~np.isnan(xvar) & ~np.isnan(yvar) & maski

    # --- define a linear function for ODR ---
    def f_lin(B, x):
        return B[0] * x + B[1]  # B[0]=slope, B[1]=intercept

    # package the data for ODR (unweighted)
    data = odr.RealData(xvar[mask], yvar[mask]) #, sy=yerr[mask]   
    # build and run ODR
    model = odr.Model(f_lin)
    odr_obj = odr.ODR(data, model, beta0=[1.0, 0.0])
    out = odr_obj.run()

    slope, intercept = out.beta

    # compute fitted line & stats
    xp = np.array(xlims)
    yp = intercept + slope * xp

    ax.fill_between(xp, xp - uwibass.PF_sampling_uncertainty, xp + uwibass.PF_sampling_uncertainty, color='grey', label='Sampling precision', alpha=.2, zorder=0)
    ax.plot(xp, xp, color='grey', linestyle='--', label='1:1 line')
    ax.plot(xp, yp, color='deepskyblue', label='ODR', linewidth=1.5)


    # R² from ODR predictions
    y_pred = intercept + slope * xvar[mask]
    ss_res = np.sum((yvar[mask] - y_pred)**2)
    ss_tot = np.sum((yvar[mask] - np.mean(yvar[mask]))**2)
    r2_odr = 1 - ss_res / ss_tot

    # RMSE between x and y (as you had it)
    RMSE = np.sqrt(np.mean((xvar[mask] - yvar[mask])**2))

    ax.set_xlabel('Magnaprobe snow depth [m]')
    ax.set_ylabel('Pathfinder, UWiBaSS snow depth [m]')

    ax.text(
        0.95, 0.05,
        f"N = {len(xvar[mask])}"
        f"\nR² = {r2_odr:.2f}"
        f"\nRMSE = {RMSE:.2f} m",
        transform=ax.transAxes, va='bottom', ha='right', size=10
    )
    

    custom_legend_elements = [
        Line2D([0], [0], lw=1.2, color='silver', markerfacecolor='dimgrey', markeredgecolor='none', marker='o', markersize=6, label='Data points + uncertainties'),
        Line2D([0], [0], lw=1.2, color='silver', markerfacecolor='red', markeredgecolor='none', marker='o', markersize=6, label='MP from internal layer'),
        Line2D([0], [0], color='none',  linestyle='none'),

        Line2D([0], [0], color='deepskyblue', lw=1.5, label='ODR'),
        Line2D([0], [0], color='grey', lw=1, linestyle='--', label='1:1 line'),
        Patch(color='grey', alpha=.2, label='Sampling precision'),
        
    ]
    ax.legend(handles=custom_legend_elements, loc='upper left', fontsize=10, frameon=False)

