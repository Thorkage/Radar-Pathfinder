import sys

from helper_functions import *
del sys.modules['helper_functions']
from helper_functions import *

from radar_functions import *
del sys.modules['radar_functions']
from radar_functions import *


def compute_cost_map(section,
                     PF_parameters,
                     first_layer='top', layer='top',
                     debug=False
                     ):
    """
    
    """


    a = PF_parameters['a']
    b = PF_parameters['b']
    c = PF_parameters['c']
    d = PF_parameters['d']
    vertical_radius = PF_parameters['jumpiness'] + 1
    sigma = PF_parameters['sigma']

    section_blurred = gaussian_filter(section, sigma=sigma)
    
    section_blurred = np.clip(section_blurred, 0, None)  # Ensure no negative values
    reflection_cost = 1.0 - NormalizeData(section_blurred)

    hxx, hxy, hyy = hessian_matrix(abs(section), sigma=sigma, order='xy')
    
    # eigvals = hessian_matrix_eigvals([hxx, hxy, hyy])[1]
    # ridge_cost = NormalizeData(eigvals + hyy)
    
    ridge_cost = NormalizeData(hyy)

    #  ? ADJUSTED FROM A PURE MAX AMPLITUDE TO A PEAK PROMINENCE BASED COST
    # TODO Still on the todo: wavelet transform based peak detection?
    peak_cost = np.zeros_like(section)

    for x in range(section.shape[1]):
        column = (section_blurred[:, x])
        peaks, _ = find_peaks(column)
        prominences = peak_prominences(column, peaks)[0]
        dominant_peaks = prominences[prominences > np.quantile(prominences, 0.9)]

        length = max(3, len(dominant_peaks))  # at least 3 peaks, or 10% of dominant peaks
        
        top_N_peaks = peaks[np.argsort(prominences)[-length:]]
        dominant_peaks = np.sort(prominences)[-length:]
                
        if first_layer != None:
            if layer == 'top':
                max_peak_idx = np.argmax(dominant_peaks)
                max_peak_value = top_N_peaks[max_peak_idx]
                candidates = np.where(top_N_peaks < max_peak_value)[0]
                
                if len(candidates) > 0:
                    idx = np.min(top_N_peaks[candidates])  
                    val = np.min(dominant_peaks[candidates])
                else:
                    idx = 0
                    val = 0
                
            elif layer == 'bottom':
                arg_idx = np.argmax(dominant_peaks)
                idx = top_N_peaks[arg_idx]
                val = np.max(dominant_peaks)
            
        elif first_layer == None:
            idx = top_N_peaks
            val = dominant_peaks
            
        peak_cost[idx, x] = val
                
            
    peak_cost = 1 - NormalizeData(peak_cost)
    continuity_cost = fast_crosscorr_continuity_cost(np.abs(section), R=vertical_radius)
    continuity_cost[:5, :] = 1.0  
    continuity_cost[-5:, :] = 1.0 
    continuity_cost = NormalizeData(continuity_cost)
    
    if layer != None:
        cost = NormalizeData((a * reflection_cost + b * ridge_cost + c * peak_cost + d * continuity_cost))
        partial_costs = {'reflection': reflection_cost, 'ridge': ridge_cost, 'peak': peak_cost, 'continuity': continuity_cost}
        
        if debug:
            return cost, partial_costs
        else:
            return cost
    else: 
        cost = NormalizeData((a * reflection_cost + b * ridge_cost))
        partial_costs = {'reflection': reflection_cost, 'ridge': ridge_cost}
        
        if debug:
            return cost, partial_costs
        else:
            return cost



def compute_next_cost_map(section,
                          PF_parameters,
                          first_layer, layer, 
                          first_reflection_cost, first_ridge_cost, first_continuity_cost,
                          debug=False
                          ):
    """
    
    """
    a = PF_parameters['a']
    b = PF_parameters['b']
    c = PF_parameters['c']
    d = PF_parameters['d']
    vertical_radius = PF_parameters['jumpiness'] + 1
    sigma = PF_parameters['sigma']
    
    
    section_blurred = gaussian_filter(section, sigma=sigma)
    reflection_cost = first_reflection_cost.copy()
    ridge_cost = first_ridge_cost.copy()

    #  ? ADJUSTED FROM A PURE MAX AMPLITUDE TO A PEAK PROMINENCE BASED COST
    # TODO Still on the todo: wavelet transform based peak detection?
    peak_cost = np.zeros_like(section)

    for x in range(section.shape[1]):
        column = (section_blurred[:, x])
        peaks, _ = find_peaks(column)
        prominences = peak_prominences(column, peaks)[0]
        dominant_peaks = prominences[prominences > np.quantile(prominences, 0.9)]

        # length = max(3, len(dominant_peaks))  # at least 3 peaks, or 10% of dominant peaks
        length = len(dominant_peaks)
        top_N_peaks = peaks[np.argsort(prominences)[-length:]]
        dominant_peaks = np.sort(prominences)[-length:]
                
        if first_layer != None:
            if layer == 'top':
                max_peak_idx = np.argmax(dominant_peaks)
                max_peak_value = top_N_peaks[max_peak_idx]
                candidates = np.where(top_N_peaks < max_peak_value)[0]
                
                if len(candidates) > 0:
                    idx = np.min(top_N_peaks[candidates])  
                    val = np.min(dominant_peaks[candidates])
                else:
                    idx = 0
                    val = 0
                
            elif layer == 'bottom':
                arg_idx = np.argmax(dominant_peaks)
                idx = top_N_peaks[arg_idx]
                val = np.max(dominant_peaks)
            
        elif first_layer == None:
            idx = top_N_peaks
            val = dominant_peaks
            
        peak_cost[idx, x] = val
                
            
    peak_cost = 1 - NormalizeData(peak_cost)
    
    continuity_cost= first_continuity_cost.copy()
    
    if layer != None:
        cost = NormalizeData((a * reflection_cost + b * ridge_cost + c * peak_cost + d * continuity_cost))
        partial_costs = {'reflection': reflection_cost, 'ridge': ridge_cost, 'peak': peak_cost, 'continuity': continuity_cost}
        
        if debug:
            return cost, partial_costs
        else:
            return cost
    else: 
        cost = NormalizeData((a * reflection_cost + b * ridge_cost))
        partial_costs = {'reflection': reflection_cost, 'ridge': ridge_cost}
        
        if debug:
            return cost, partial_costs
        else:
            return cost
    


def fast_crosscorr_continuity_cost(section, R):

    """
    Compute a continuity cost term based on local shift‐based cross‐correlation.
    
    For each pixel (x,y), we look at the window of height (2R+1) around (x,y) in column x,
    and we compare it against the same‐sized window in column x-1 under vertical shifts
    dx ∈ [−R..R]. We compute the maximum normalized cross‐correlation over those shifts,
    then set
   
        cost[y, x] = 1 − max_corr[y, x]
   
    so that a pixel whose local patch matches (up to a small vertical shift) a patch in
    the previous column has low cost. For x=0, cost[:,0] remains 1.0.
    
    Parameters
    ----------
    section : 2D np.ndarray, shape (H, W)
        Input echogram (float). section[y, x] is amplitude at row y, column x.
    R : int
        Vertical radius for sliding window and allowed shifts. The window is 2R+1 rows tall.
    
    Returns
    -------
    cost : 2D np.ndarray, shape (H, W), dtype=float32
        Continuity‐based cost ∈ [0,1]. Lower cost = stronger local continuity.
    """
    H, W = section.shape
    kernel = np.ones(2 * R + 1, dtype=np.float32)  # 1D kernel for moving sum
    
    cost = np.ones((H, W), dtype=np.float32)
    eps = 1e-8  # to avoid divide-by-zero
    
    for x in range(1, W):
        col = section[:, x]
        prev = section[:, x - 1]
        
        # Precompute squared columns
        col_sq = col * col
        prev_sq = prev * prev
        
        # 1) Moving sum of col_sq with a (2R+1)-long window
        sum_col_sq = convolve(col_sq, kernel, mode='reflect')
        # 2) Moving sum of prev_sq for the unshifted 'prev' column
        sum_prev_sq_base = convolve(prev_sq, kernel, mode='reflect')
        
        best_corr = np.zeros(H, dtype=np.float32)  # track max over dx
        
        for dx in range(-R, R + 1):
            # 3) Shift prev by dx (circularly)
            prev_shifted = np.roll(prev, dx)
            # prev_shifted_sq = np.roll(prev_sq, dx)
            
            # 4) The moving sum of prev_shifted_sq is just sum_prev_sq_base rolled by dx
            sum_prev_sq = np.roll(sum_prev_sq_base, dx)
            
            # 5) Compute elementwise product of col * prev_shifted, then moving sum
            prod = col * prev_shifted
            sum_prod = convolve(prod, kernel, mode='reflect')
            
            # 6) Normalized correlation = sum_prod / (||col_window|| * ||prev_shifted_window||)
            denom = np.sqrt((sum_col_sq + eps) * (sum_prev_sq + eps))
            corr = sum_prod / denom
            
            # 7) Clamp negative correlations to zero
            corr[corr < 0] = 0
            
            # 8) Keep the maximum correlation across all shifts dx
            np.maximum(best_corr, corr, out=best_corr)
        
        # 9) Finally, cost = 1 − best_corr
        cost[:, x] = 1.0 - (best_corr)
    
    # x=0: no previous column → cost remains 1.0
    return cost


def make_penalty_matrix(height, penalty=2.0):
    """
    
    """

    y = np.arange(height)
    return penalty * (y[:, None] - y[None, :])**2

# USING THE HARD CONSTRAINT IN THE VERTICAL
# @njit(parallel=True, cache=True)
# def fast_dp_forward(cost, vertical_radius=3):
#     height, width = cost.shape
#     cumulative_cost = np.full((height, width), np.inf, dtype=np.float32)
#     backtrack = np.zeros((height, width), dtype=np.int32)

#     cumulative_cost[:, 0] = cost[:, 0]

#     for x in range(1, width):
#         for yy in prange(height):
#             best_total = np.inf
#             best_prev = 0
#             yymin = max(0, yy - vertical_radius)
#             yymax = min(height, yy + vertical_radius + 1)
#             for y_prev in range(yymin, yymax):
#                 # penalty = penalty_matrix[yy, y_prev]
#                 total = cost[yy, x] + cumulative_cost[y_prev, x - 1]
#                 if total < best_total:
#                     best_total = total
#                     best_prev = y_prev
#             cumulative_cost[yy, x] = best_total
#             backtrack[yy, x] = best_prev

#     return cumulative_cost, backtrack

# USING THE SOFT PENALTY IN THE VERTICAL
@njit(parallel=True, cache=True)
def fast_dp_forward(cost, penalty_matrix):
    """
    
    """
    
    height, width = cost.shape
    cumulative_cost = np.full((height, width), np.inf, dtype=np.float32)
    backtrack = np.zeros((height, width), dtype=np.int32)

    cumulative_cost[:, 0] = cost[:, 0]

    for x in range(1, width):
        for yy in range(height):
            best_total = np.inf
            best_prev = 0
            for y_prev in range(height):
                penalty = penalty_matrix[yy, y_prev]
                total = cost[yy, x] + cumulative_cost[y_prev, x - 1] + penalty
                if total < best_total:
                    best_total = total
                    best_prev = y_prev
            cumulative_cost[yy, x] = best_total
            backtrack[yy, x] = best_prev

    return cumulative_cost, backtrack


def backtrack_path(cumulative_cost, backtrack):
    """
    
    """
    
    height, width = cumulative_cost.shape
    path = []
    y = np.argmin(cumulative_cost[:, -1])
    for x in reversed(range(width)):
        path.append((y, x))
        y = backtrack[y, x]
    path.reverse()
    return path


# @njit(parallel=True, cache=True)
# def fast_dp_backward(cost, vertical_radius):
#     height, width = cost.shape
#     cumulative_cost = np.full((height, width), np.inf, dtype=np.float32)
#     forwardtrack = np.zeros((height, width), dtype=np.int32)

#     cumulative_cost[:, -1] = cost[:, -1]

#     for x in range(width - 2, -1, -1):
#         for yy in prange(height):
#             best_total = np.inf
#             best_next = 0
#             yymin = max(0, yy - vertical_radius)
#             yymax = min(height, yy + vertical_radius + 1)
#             for y_next in range(yymin, yymax):
#                 # penalty = penalty_matrix[yy, y_next]
#                 total = cost[yy, x] + cumulative_cost[y_next, x + 1] #+ penalty
#                 if total < best_total:
#                     best_total = total
#                     best_next = y_next
#             cumulative_cost[yy, x] = best_total
#             forwardtrack[yy, x] = best_next

#     return cumulative_cost, forwardtrack

# def backtrack_path_backward(cumulative_cost, forwardtrack):
#     height, width = cumulative_cost.shape
#     path = []
#     y = np.argmin(cumulative_cost[:, 0])
#     for x in range(width):
#         path.append((y, x))
#         y = forwardtrack[y, x]
#     return path

def find_optimal_path(cost, PF_parameters, direction="forward", layer="bottom"):
    """
    
    """
    
    vertical_radius = PF_parameters['jumpiness']
    bias_strength = PF_parameters['db']
    
    h, w = cost.shape
    bias = np.linspace(0, 1, h).reshape(-1, 1)  # top = 0, bottom = 1

    if layer == "top":
        cost = cost + bias_strength * bias  # penalize deeper paths
    elif layer == "bottom":
        cost = cost + bias_strength * (1 - bias)  # penalize shallow paths

    penalty_matrix = make_penalty_matrix(h, penalty=0.05 / (vertical_radius**2))

    if direction == "forward":
        cumulative_cost, backtrack = fast_dp_forward(cost, penalty_matrix)
        path = backtrack_path(cumulative_cost, backtrack)
        
    # elif direction == "backward":
    #     cumulative_cost, forwardtrack = fast_dp_backward(cost, penalty_matrix)
    #     path = backtrack_path_backward(cumulative_cost, forwardtrack)
    # else:
    #     raise ValueError("Invalid direction. Use 'forward' or 'backward'.")

    path_signal = [cost[y, x] for y, x in path]
    return path, path_signal




    
    



#PATH DISENTANGLER
def find_switch_regions(diff, window_size=20):
    """
    Find contiguous regions where diff <= 0.
    Each region is buffered by window_size and overlapping buffers are merged.
    """
    below_zero = np.where(diff <= 0)[0]
    if len(below_zero) == 0:
        return []

    # Find contiguous runs
    regions = []
    start = below_zero[0]
    for i in range(1, len(below_zero)):
        if below_zero[i] != below_zero[i - 1] + 1:
            regions.append((start, below_zero[i - 1]))
            start = below_zero[i]
    regions.append((start, below_zero[-1]))

    # Buffer and merge
    buffered = [(max(0, s - window_size), min(len(diff), e + window_size)) for s, e in regions]
    merged = [buffered[0]]
    for curr in buffered[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], curr[1]))
        else:
            merged.append(curr)

    return merged


def refine_intersections(paths, section, cost, PF_parameters, window_size=20, n_switches=10000):
    """
    
    
    """

    path_arr = np.column_stack([l for l in paths.values()])
    
    refined_paths = copy.deepcopy(paths)
    
    cost_tmp = copy.deepcopy(cost)
    
    diff = np.array(paths['bottom'])-np.array(paths['top'])
    merged = find_switch_regions(diff, window_size=window_size)
    
    # print(len(merged), n_switches)
    
    if len(merged) >= n_switches or len(merged) == 0:
        
        window_size -= 5
        if window_size <= 0 or len(merged) >= n_switches:
            # print('converged')
            
            # After convergence: fix paths hitting boundaries
            top_path = refined_paths['top']
            bottom_path = refined_paths['bottom']
            
            for x in range(len(top_path)):
                if top_path[x] <= 30:  # too close to top
                    top_path[x] = bottom_path[x]
                if bottom_path[x] >= section.shape[0] - 30:  # too close to bottom
                    bottom_path[x] = top_path[x]

            return refined_paths

        return refine_intersections(refined_paths, section, cost, PF_parameters, window_size=window_size, n_switches=len(merged))
    

    for region in merged:

        start = region[0]
        end = region[1]
        
        for layer in paths.keys():
            
            if layer == 'top':
                y_start = np.min(path_arr[start, :])
                y_end = np.min(path_arr[end-1, :])
                path_reduc = np.max(path_arr[start:end,:], axis=1)
                
            if layer == 'bottom':
                y_start = np.max(path_arr[start, :])
                y_end = np.max(path_arr[end-1, :])
                path_reduc = np.min(path_arr[start:end,:], axis=1)

            sub_cost = cost_tmp[:, start:end]
            sub_cost[y_start, 0] = -1
            sub_cost[y_end, -1] = -1
            
            if np.mean(abs(diff[start+window_size:end-window_size])) > 5:
                masking_strengths = np.zeros(np.shape(range(end-start)))
                masking_strengths[diff[start:end] < 0] = 1
                sub_cost = mask_path_in_cost2(sub_cost, zip(path_reduc, range(end-start)), radius=3, strength=masking_strengths)

            new_path, _ = find_optimal_path(sub_cost, PF_parameters)
            refined_y = np.array([y for y, _ in new_path])

            refined_paths[layer][start:end] = refined_y
            back_path = list(zip(refined_y, range(start, end)))
            
            masking_strengths = np.zeros(len(diff))
            if np.mean(np.mean(abs(diff[start:start+window_size])) + np.mean(abs(diff[end-window_size:end]))) > 20:
                masking_strengths[diff >= 0] = 1
                
            cost_tmp = mask_path_in_cost2(cost, back_path, radius=3, strength=masking_strengths)



    return refine_intersections(refined_paths, section, cost, PF_parameters, window_size=window_size, n_switches=len(merged))





# INTERNAL LAYERS
def compute_internal_layers(RADAR,
                            PF_parameters,
                            minimum_cost=0.65
                            ):
    """
    Computes the internal layers of the RADAR object.
    """
    
    print('Computing internal layers... ')
    
    # Flatten the section to the top interface
    flattened_section = flatten_to_interface(RADAR.rx_rpca, RADAR,
                        interface='top',
                        smoothing_window=3,
                        reduce=False,
                        top_buffer=None
                        )
        
    third_cost = compute_cost_map(Quickboost(flattened_section),
                                  PF_parameters,
                                  layer=None,
                                  first_layer=None,
                                  debug=False
                                  )
    flattened_bottom_path = pd.Series(RADAR.PF_bottom_interface).rolling(window=3, min_periods=0, center=True).mean().astype(int) - pd.Series(RADAR.PF_top_interface).rolling(window=3, min_periods=0, center=True).mean().astype(int)

    third_cost = mask_path_in_cost(third_cost,
                                zip([0]*len(RADAR.PF_top_interface), range(len(RADAR.PF_top_interface))),
                                radius=2, #TODO: BASE THIS ON PHYSICS (Expected range resolution)
                                strength=np.inf
                                )

    third_cost = mask_path_in_cost(third_cost,
                                zip(flattened_bottom_path, range(len(RADAR.PF_bottom_interface))),
                                radius=2, #TODO: BASE THIS ON PHYSICS (Expected range resolution)
                                strength=np.inf
                                )

    third_cost = mask_cost_below_above_path(third_cost,
                                            zip(flattened_bottom_path, range(len(RADAR.PF_bottom_interface))),
                                            np.inf,
                                            layer='bottom'
                                            )

    third_cost_stored = third_cost.copy()
    tmp = third_cost_stored.copy()
    tmp[np.isinf(tmp)] = np.nan


    seeds, cost_at_seeds = _find_seeds(tmp)
    ri, ci = seeds[:,1], seeds[:,0]
    ri = ri.tolist()
    ci = ci.tolist()

    struct = np.zeros(tmp.shape)
    struct_2 = np.zeros(tmp.shape)

    for i in range(len(ri)):
        struct[ri[i], ci[i]] = 1
        struct_2[ri[i], ci[i]] = cost_at_seeds[i]
        
    labeled_array, label_regions, label_areas, label_mean_costs, label_scores = _label_array(struct, tmp)
    paths, path_costs, path_SNR, path_lengths, idx_from_bottom_interface, idx_from_top_interface = pathfind_in_regions(labeled_array,
                                                                                                            tmp,
                                                                                                            RADAR.rx_rpca,
                                                                                                            RADAR,
                                                                                                            buffer=2,
                                                                                                            # min_cost=0.75,
                                                                                                            filter=False,
                                                                                                            )                                                                                

    min_SNR = 1
    min_cost = minimum_cost

    filtered_paths = [path for path, SNR, cost in zip(paths, path_SNR, path_costs) if SNR > min_SNR and cost < min_cost]
    filtered_path_costs = [cost for cost, SNR in zip(path_costs, path_SNR) if SNR > min_SNR and cost < min_cost]
    filtered_path_SNR = [snr for snr, cost in zip(path_SNR, path_costs) if snr > min_SNR and cost < min_cost]

    RADAR.PF_internal_layers = {}
    RADAR.PF_internal_layers['paths'] = filtered_paths.copy()
    RADAR.PF_internal_layers['costs'] = filtered_path_costs.copy()
    RADAR.PF_internal_layers['SNR'] = filtered_path_SNR.copy()

    idx_from_top = []
    idx_from_bottom = []
    relative_idx = []
    for p in RADAR.PF_internal_layers['paths']:
        py, px = list(zip(*p))
        py = np.array(py)
        px = np.array(px)
        idx_from_top.append(py) 
        idx_from_bottom.append(RADAR.PF_bottom_interface[px] - RADAR.PF_top_interface[px] - py) 
        relative_idx.append(py / (RADAR.PF_bottom_interface[px] - RADAR.PF_top_interface[px]))
    idx_from_top = np.concatenate(idx_from_top)
    idx_from_bottom = np.concatenate(idx_from_bottom)
    relative_idx = np.concatenate(relative_idx)


    RADAR.PF_internal_layers['idx_from_top'] = idx_from_top
    RADAR.PF_internal_layers['idx_from_bottom'] = idx_from_bottom
    RADAR.PF_internal_layers['relative_idx'] = relative_idx

    return RADAR



def _find_seeds(cost):
    h,w = cost.shape
    seeds = []
    cost_at_seeds = []
    for x in range(w):
        col = cost[:,x]
        locs = argrelextrema(col, np.less)[0]
        
        # locs = argrelextrema(col, np.greater)[0]
        # locs = np.concatenate((locs_min, locs_max))
        for y in locs:
            seeds.append((x,y))
            cost_at_seeds.append(cost[y,x])
            
    seeds = np.array(seeds)
    cost_at_seeds = np.array(cost_at_seeds)
    return seeds, cost_at_seeds
    
    
def _label_array(struct, tmp):
    labeled_array = label(struct, connectivity=2)
    regions = regionprops(labeled_array)
    areas = NormalizeData([reg.area for reg in regions])
    mean_costs = 1 - NormalizeData(ndimage.median(tmp, labels=labeled_array, index=np.arange(1, len(regions)+1)))
    scores =  mean_costs ** areas
    
    return labeled_array, regions, areas, mean_costs, scores


# TODO: can we parallelize this? (pathfinding itself is already parallelized, maybe)
def pathfind_in_regions(labeled_array, tmp, section, RADAR, buffer=2, min_cost=None, filter=False):
    
    props = regionprops(labeled_array)
    paths = []
    path_costs = []
    path_SNR = []
    idx_from_bottom_interface = []
    idx_from_top_interface = []
    
     # noise
    flattened_section = flatten_to_interface(section, RADAR,
                        interface='top',
                        smoothing_window=3,
                        reduce=False
                        )

    flattened_bottom_trace = pd.Series(RADAR.PF_bottom_interface).rolling(window=3, min_periods=0, center=True).mean().astype(int) - pd.Series(RADAR.PF_top_interface).rolling(window=3, min_periods=0, center=True).mean().astype(int)

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
    global_noise_snow = np.std(snow_signal)
    
    for region in tqdm(props):
        coords = region.coords
        _, x_coords = zip(*coords)

        x_start = max(0, min(x_coords) - buffer)
        x_end = min(tmp.shape[1], max(x_coords) + buffer + 1)

        local_cost = tmp[:, x_start:x_end]
        # local_section = section[:, x_start:x_end]

        region_mask = np.zeros_like(local_cost, dtype=bool)
        for y, x in coords:
            if x_start <= x < x_end:
                region_mask[y, x - x_start] = True

        penalty_mask = np.ones_like(local_cost)
        penalty_mask[region_mask] = 0  # no penalty in original region
        masked_cost = local_cost + penalty_mask  # small penalty outside

        left_candidates = [y for y, x in coords if x == x_start + buffer]
        right_candidates = [y for y, x in coords if x == x_end - buffer - 1]


        if not left_candidates or not right_candidates:
            continue

        y_start = min(left_candidates)
        y_end = min(right_candidates)
        masked_cost[y_start, 0] -= -.5
        masked_cost[y_end, -1] -= -.5

        path, path_cost = find_optimal_path(masked_cost, RADAR.PF_parameters, layer=None)
        
        subset_path = [pt for pt in path if buffer <= pt[1] < masked_cost.shape[0] - buffer]

        global_path = [(y, x + x_start) for y, x in subset_path]
        ys = [np.array([y for y, x in global_path])]
        xs = [np.array([x for y, x in global_path])]
        path_P = flattened_section[ys, xs]        
        
        paths.append(global_path)
        path_costs.append(np.median(path_cost))
        path_SNR.append(np.median(path_P / global_noise_snow))

    path_lengths = [len(path) for path in paths]
    
   
    if min_cost is None:
        min_cost = np.nanquantile(path_costs, 0.5)
        
    # if filter == True:
    #     # TODO: use SNR to filter paths that are not stron?
    #     print(f"Found {len(paths)} paths with mean cost < {min_cost:.3f}. Before filtering: {len(paths)} paths.")
    #     paths = [path for path, cost in zip(paths, path_costs) if cost < min_cost]
    #     path_lengths = [length for length, cost in zip(path_lengths, path_costs) if cost < min_cost]
    #     idx_from_bottom_interface = [idx for idx, cost in zip(idx_from_bottom_interface, path_costs) if cost < min_cost]
    #     idx_from_top_interface = [idx for idx, cost in zip(idx_from_top_interface, path_costs) if cost < min_cost]
    #     path_costs = [cost for cost in path_costs if cost < min_cost]

    return paths, path_costs, path_SNR, path_lengths, idx_from_bottom_interface, idx_from_top_interface






# def rolling_std(df, lengthscale):
#     coords = df[['x', 'y']].values
#     roughness = np.full(len(df), np.nan)
#     for i, (xi, yi) in enumerate(coords):
#         dists = np.sqrt((coords[:, 0] - xi)**2 + (coords[:, 1] - yi)**2)
#         mask = dists <= lengthscale / 2
#         if np.sum(mask) > 3:
#             roughness[i] = df['elev'][mask].std()
#     return roughness

# def calculate_topo_roughness(x, y, elev, lengthscale=0.5):
#     """
#     Calculate surface roughness as the std of elevation over a moving window
#     of given lengthscale (in meters) using (x, y) coordinates.
#     """
#     df = pd.DataFrame({'x': np.array(x), 'y': np.array(y), 'elev': elev})
#     return rolling_std(df, lengthscale)


# def group_peaks_into_regions(peaks_list, window_size=10):

#     markers = []
#     for name, arr in peaks_list:
#         for idx in arr:
#             markers.append((idx, name))
#     markers.sort()

#     regions = []
#     used = np.zeros(len(markers), dtype=bool)
#     for i, (idx, _) in enumerate(markers):
#         if used[i]:
#             continue
#         region_start = idx - window_size
#         region_end = idx + window_size
#         region_indices = []
#         region_names = set()
#         region_markers = []
#         for j, (idx2, name2) in enumerate(markers):
#             if region_start <= idx2 <= region_end:
#                 used[j] = True
#                 region_indices.append(idx2)
#                 region_names.add(name2)
#                 region_markers.append((idx2, name2))
#         regions.append({
#             'start': region_start,
#             'end': region_end,
#             'indices': sorted(region_indices),
#             'diversity': len(region_names),
#             'count': len(region_indices),
#             'combined_score' :len(region_indices) + len(region_names),
#             'markers': region_markers
#         })


#     if not regions:
#         return []
#     regions.sort(key=lambda r: r['start'])
#     merged = [regions[0]]
#     for reg in regions[1:]:
#         last = merged[-1]
#         if reg['start'] <= last['end']:
#             # Merge
#             last['end'] = max(last['end'], reg['end'])
#             last['indices'] = sorted(set(last['indices']).union(reg['indices']))
#             last['diversity'] = len(set([m[1] for m in last['markers']] + [m[1] for m in reg['markers']]))
#             last['count'] = len(set(last['indices']))
#             last['combined_score'] = last['count'] + last['diversity']
#             last['markers'].extend(reg['markers'])
#         else:
#             merged.append(reg)

#     filtered = [
#         r for r in merged
#         if not ((r['end'] - r['start']) <= window_size)
#     ]
    
#     filtered.sort(key=lambda r: (r['diversity'], r['count']), reverse=True)
#     return filtered

# def find_flag_regions(section, path):
    
#     n_cols = section.shape[1]
#     all_path_prominences    = np.empty(n_cols)

#     for i in range(n_cols):
#         x = section[:, i]
#         idx  = path[i]
#         all_path_prominences[i]    = np.max(peak_prominences(x, [max(0, idx-1), idx, min(len(x)-1, idx+1)])[0])
        
#     top_min_prom = pd.Series(all_path_prominences)
#     peaks_prom_low = find_peaks(1-top_min_prom, height=np.quantile(1-top_min_prom, .98))[0]
    
#     top_diff_prom = pd.Series(np.diff(top_min_prom, prepend=top_min_prom[0]))
#     peaks_prom_diff = find_peaks(np.abs(top_diff_prom), height=np.quantile(np.abs(top_diff_prom), .98))[0]
    
#     # path_diff= pd.Series(np.diff(refined_paths['top'], prepend=refined_paths['top'][0]))
#     # peaks_path_diff = find_peaks(np.abs(path_diff), height=np.quantile(np.abs(path_diff), .99))[0]
    
#     section_along_path = np.diff(section[path, np.arange(n_cols)], prepend=top_min_prom[0])
#     peaks_section = find_peaks(np.abs(section_along_path), height=np.quantile(np.abs(section_along_path), .98))[0] 

    
#     # Example usage:
#     peaks_list = [
#         ('prom_low', peaks_prom_low),
#         ('prom_diff', peaks_prom_diff),
#         ('section', peaks_section)
#     ]
#     regions = group_peaks_into_regions(peaks_list,)
#     scores = np.array([reg['combined_score'] for reg in regions])
#     length = len(scores[scores > np.quantile(scores, 0.85)])
    
    
#     return regions[:length]
















# def find_optimal_path(cost, section, vertical_radius=3, layer="top", allow_wrap=False, wrap_penalty=2.0, bias_strength=0.1):
    
#     h, w = cost.shape
#     bias = np.linspace(0, 1, h).reshape(-1, 1) 
    
#     if layer == "top":
#         cost = cost + bias_strength * bias  # penalize deeper paths
#     elif layer == "bottom":
#         cost = cost + bias_strength * (1 - bias)  # penalize shallow paths
        
#     penalty_matrix = make_penalty_matrix(cost.shape[0], penalty=0.05 / (vertical_radius**2))

#     if allow_wrap:
#         cumulative_cost, backtrack = fast_dp_forward_wrapping(cost, penalty_matrix, wrap_penalty=wrap_penalty)
#     else:
#         cumulative_cost, backtrack = fast_dp_forward(cost, penalty_matrix)

#     path = backtrack_path(cumulative_cost, backtrack)
#     path_signal = [abs(section[y, x]) for y, x in path]

#     return path, path_signal
# EXPERIMENTING WITH WRAPPING 
# @njit(parallel=True, cache=True)
# def fast_dp_forward_wrapping(cost, penalty_matrix, wrap_penalty=1.0):
#     """
#     Dynamic programming pathfinding with vertical wrapping support.
    
#     Parameters:
#         cost (2D np.ndarray): Cost map of shape (H, W)
#         penalty (float): base penalty scaling factor
#         wrap_penalty (float): additional penalty when wrapping occurs
    
#     Returns:
#         cumulative_cost (2D np.ndarray)
#         backtrack (2D np.ndarray)
#     """
#     height, width = cost.shape
#     cumulative_cost = np.full((height, width), np.inf, dtype=np.float32)
#     backtrack = np.zeros((height, width), dtype=np.int32)

#     cumulative_cost[:, 0] = cost[:, 0]

#     for x in range(1, width):
#         for yy in prange(height):  # current y
#             best_total = np.inf
#             best_prev = 0
#             for y_prev in range(height):  # previous y
#                 dy = abs(yy - y_prev)
#                 wrapped_dy = height - dy
#                 used_dy = min(dy, wrapped_dy)
#                 wrap = dy != used_dy

#                 # penalty_cost = penalty * (used_dy ** 2)
#                 penalty = penalty_matrix[yy, y_prev]
#                 if wrap:
#                     penalty += wrap_penalty

#                 total = cost[yy, x] + cumulative_cost[y_prev, x - 1] + penalty

#                 if total < best_total:
#                     best_total = total
#                     best_prev = y_prev
#             cumulative_cost[yy, x] = best_total
#             backtrack[yy, x] = best_prev

#     return cumulative_cost, backtrack