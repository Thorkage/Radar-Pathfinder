from helper_functions import *
del sys.modules['helper_functions']
from helper_functions import *

from radar_functions import *
del sys.modules['radar_functions']
from radar_functions import *

from pathfinder import *
del sys.modules['pathfinder']
from pathfinder import *


class LayerCorrectionTool:
    def __init__(self,
                 RADAR,
                 window_size=50, chunk_size=700, overlap=128, #displaying options
                 has_flightstate=True,
                 has_geolocation=True,
                 add_grounddata=False, df_MP=None, df_SP=None, dict_SP=None
                 ):

        self.add_grounddata = add_grounddata
        self.has_flightstate = has_flightstate
        self.has_geolocation = has_geolocation
        self.df_MP = df_MP
        self.df_SP = df_SP
        self.dict_SP = dict_SP
        self.RADAR = RADAR
        
        if RADAR.radar_type == 'UWiBaSS':
            self.full_path = os.path.join(RADAR.data_path, 'UWiBaSS', RADAR.datasetID)
        else:
            self.full_path = None
            
        self.section = RADAR.rx_rpca
        self.cost_top = RADAR.PF_cost_maps['top']
        self.cost_bottom = RADAR.PF_cost_maps['bottom']
        self.unbiased_cost = RADAR.PF_cost_maps['unbiased'].copy()
        self.internal_paths = RADAR.PF_internal_layers.get('paths', None) if RADAR.PF_internal_layers else None
        self.internal_costs = RADAR.PF_internal_layers.get('costs', None) if RADAR.PF_internal_layers else None
        self.internal_SNR = RADAR.PF_internal_layers.get('SNR', None) if RADAR.PF_internal_layers else None
        # self.regions = regions
        # self.base_len_regions_top = len(regions['top'])
        # self.base_len_regions_bottom = len(regions['bottom'])
        self.base_cost_top = self.unbiased_cost.copy()  # unmasked base
        self.base_cost_bottom = self.unbiased_cost.copy()  # unmasked base
        self.path_top = RADAR.PF_top_interface.copy()
        self.path_bottom = RADAR.PF_bottom_interface.copy()
        self.window_size = window_size
        self.jumpiness = RADAR.PF_parameters['jumpiness']
        self.clicked_points = {'top': [], 'bottom': []}
        self.all_clicked_points = {'top': [], 'bottom': []}  # Persistent storage
        
        self.recompute_counter = {
            'top': np.zeros(RADAR.rx_rpca.shape[1], dtype=int),
            'bottom': np.zeros(RADAR.rx_rpca.shape[1], dtype=int)
        }
        self.active_layer = 'top'
        self.dragging = False
        self.last_mouse_pos = None
        self.paths_visible = True
        self.show_internal = False
        self.use_quickboost = False
        self.space_pressed = False

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.current_chunk = 0
        self.num_chunks = int(np.ceil(RADAR.rx_rpca.shape[1] / (chunk_size - overlap)))

        self.fig = plt.figure(figsize=(20, 10), dpi=100, constrained_layout=True)

        
        
        self.figtitle = self.fig.suptitle(f'{self.RADAR.radar_type}',
                          fontsize=11, ha='left', x=0.05)
        
        gs = gridspec.GridSpec(3, 2, width_ratios=[30, 1], height_ratios=[.2, 1,  3])
        
        # plt.subplots_adjust(left=0.05, right=0.85, top=1, bottom=0.05)
        if self.has_flightstate:
            self.ax_var = self.fig.add_subplot(gs[0, 0])
            self.ax_var.set_xlim(0, self.section.shape[1])
            self.ax_var.grid(ls=':', axis='y')
            self.ax_var.spines['top'].set_color('white')
            self.ax_var.spines['right'].set_color('white')
            
            self.twinx_var = self.ax_var.twinx()
            # self.twinx_var.grid(ls=':', axis='y')
            self.twinx_var.spines['top'].set_color('white')
            self.twinx_var.spines['right'].set_color('white')
            self.var1, = self.ax_var.plot([], [], color='salmon', lw=2, alpha=0.8)
            self.var2, = self.ax_var.plot([], [], color='mediumseagreen', lw=2, alpha=0.8)
            self.var3, = self.twinx_var.plot([], [], color='red', lw=1, alpha=0.8, zorder=100)
        
        # self.var1_inset_axes = self.ax_var.inset_axes([0.95, 1, 0.05, 0.1], transform=self.ax_var.transAxes)
        # self.var2_inset_axes = self.ax_var.inset_axes([0.88, 1, 0.05, 0.1], transform=self.ax_var.transAxes)

        self.ax_overview = self.fig.add_subplot(gs[1, 0])
        self.ax_overview.set_xlim(0, self.section.shape[1])
        self.ax_overview.set_ylim(self.section.shape[0], 0)
        self.ax_overview.axis('off')
        self.ax_overview.plot(range(self.section.shape[1]), [-100] * self.section.shape[1], color='grey', lw=0.4, zorder=1, clip_on=False)
        
        self.ax = self.fig.add_subplot(gs[2, 0])
        control_ax = self.fig.add_subplot(gs[:, 1])
        control_ax.axis('off')

        self.chunk_start, self.chunk_end = self.get_chunk_bounds()
        self.im = self.ax.imshow(np.zeros((self.section.shape[0], self.chunk_end - self.chunk_start)), cmap=cmr.neutral, aspect=1/2, interpolation='none')
                
        self.line_top, = self.ax.plot([], [], color='deepskyblue', lw=2, alpha=0.3)
        self.line_bottom, = self.ax.plot([], [], color='magenta', lw=2, alpha=0.3)
        
        if self.has_geolocation:
            self.map_ax = self.fig.add_axes([0.77, 0.7, 0.2, 0.25],
                                            projection=ccrs.UTM(self.RADAR.UTM_zone, southern_hemisphere=False))
            for spine in self.map_ax.spines.values():
                spine.set_visible(False)
            self.map_ax.tick_params(axis='both', colors='white')
            self.map_ax.set_xticklabels([])
            self.map_ax.set_yticklabels([])
            
            # Show only the grid
            self.map_ax.gridlines(ls=':', lw=0.5, color='gray', alpha=0.5, visible=True)

            self.map_ax.scatter(self.RADAR.UTM_x, self.RADAR.UTM_y,
                                transform=ccrs.UTM(self.RADAR.UTM_zone, southern_hemisphere=False),
                                zorder=2,
                                s=.1,
                                color='black'
                                )

            self.map_chunk, = self.map_ax.plot(
                self.RADAR.UTM_x[self.chunk_start:self.chunk_end],
                self.RADAR.UTM_y[self.chunk_start:self.chunk_end],
                color='deepskyblue', lw=3, alpha=.5, zorder=1,
                transform=ccrs.UTM(self.RADAR.UTM_zone, southern_hemisphere=False)
            )
            self.map_xlims = self.map_ax.get_xlim()
            self.map_ylims = self.map_ax.get_ylim()
            self.map_ax.set_xlim(self.map_xlims[0] - 50, self.map_xlims[1] + 50)
            self.map_ax.set_ylim(self.map_ylims[0] - 50, self.map_ylims[1] + 50)

            if self.RADAR.UTM_zone == 33: # add toposvalbard aearial map if dataset is from Svali
                wmts_tiles = self.map_ax.add_wmts(url, layer)

        self.point_artists = []
        self.internal_artists = []
        
        # self.norm_internal = mcolors.Normalize(vmin=np.nanquantile(self.internal_costs, .05), vmax=np.nanquantile(self.internal_costs, .95))
        self.norm_internal = mcolors.Normalize(vmin=1, vmax=3)

        colors = [(188/255, 232/255, 67/255), (114/255, 114/255, 114/255)]
        cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=100)
        self.cmap_internal = cm

        self.radio_ax = self.fig.add_axes([0.83, 0.54, 0.1, 0.12])
        self.layer_labels = ['top','both','bottom']

        self.radio = RadioButtons(self.radio_ax, self.layer_labels)
 
        self.button_ax = self.fig.add_axes([0.83, 0.48, 0.1, 0.05])
        self.button = Button(self.button_ax, 'Rerun '+ u'$\u21A9$')
        
        self.reset_ax = self.fig.add_axes([0.83, 0.42, 0.1, 0.05])
        self.reset_button = Button(self.reset_ax, 'Reset points')
        
        self.toggle_ax = self.fig.add_axes([0.83, 0.36, 0.1, 0.05])
        self.toggle_button = CheckButtons(self.toggle_ax, ['Paths'], [True])
        
        if self.add_grounddata and self.has_geolocation:
            self.toggle_ax2 = self.fig.add_axes([0.83, 0.30, 0.1, 0.05])
            self.toggle_button2 = CheckButtons(self.toggle_ax2, ['In-situ'], [True])
            self.toggle_button2.on_clicked(self.toggle_insitu)
            self.show_insitu = True
        
            self.mp_scatter = self.ax.scatter([], [], s=1, color='red', marker='_', alpha=1)
            self.mp_scatter_overview = self.ax_overview.scatter([], [], s=2, color='red', zorder=100, clip_on=False)
            
            if self.df_SP is not None:
                self.snow_profile = self.ax.scatter([], [], s=1, lw=1, marker='.', color='goldenrod', alpha=.75)
                self.snow_profile_overview = self.ax_overview.scatter([], [], s=2, color='goldenrod', zorder=150, clip_on=False)
                self.map_ax.scatter(self.df_SP['UTM_x'], self.df_SP['UTM_y'],
                                                s=10, color='goldenrod', edgecolor='darkgoldenrod', marker='*',
                                                label='Snow profile',
                                                transform=ccrs.UTM(self.RADAR.UTM_zone, southern_hemisphere=False)
                                                )
                self.map_ax.set_title(f'{self.dict_SP["name"]}')

            
            self.map_ax.scatter(self.df_MP['UTM_x'], self.df_MP['UTM_y'],
                                s=1, color='red', alpha=0.5,
                                label='MagnaProbe',
                                transform=ccrs.UTM(self.RADAR.UTM_zone, southern_hemisphere=False)
                                )
            
            # self.figtitle.set_text(
            #     f'Dataset: {self.uwibass.dataset_name} - dataflashlog: {self.uwibass.dataflashlog_path.split("/")[-1].split("_")[0]}\n{self.dict_snowprofile["name"]} '
            # )
            
            
        # self.map_ax.legend(frameon=False)
        self.quickboost_toggle_ax = self.fig.add_axes([0.83, 0.24, 0.1, 0.05])
        self.quickboost_toggle = CheckButtons(self.quickboost_toggle_ax, ['Quickboost'], [False])


        self.internal_toggle_ax = self.fig.add_axes([0.83, 0.18, 0.1, 0.05])
        self.internal_toggle_button = CheckButtons(self.internal_toggle_ax, ['Internal layers'], [False])
        
        self.status_text = self.fig.text(0.83, 0.11, '', fontsize=10, fontweight='bold', color='red')

        # self.prev_ax = self.fig.add_axes([0.40, 0.9, 0.10, 0.05])
        # self.next_ax = self.fig.add_axes([0.52, 0.9, 0.10, 0.05])
        # self.prev_button = Button(self.prev_ax, '← Previous')
        # self.next_button = Button(self.next_ax, 'Next →')

        # self.chunk_text = self.fig.text(
        #     0.50, 0.87,
        #     '',  
        #     ha='center',
        #     fontsize=10
        #     )
        
        # self.top_regions_text = self.fig.text(
        #     0.88, 0.92,
        #     f'{len(self.regions["top"])}/{self.base_len_regions_top} notifications in top layer',  
        #     ha='center',
        #     fontsize=10
        #     )
        
        # self.bottom_regions_text = self.fig.text(
        #     0.88, 0.88,
        #     f'{len(self.regions["bottom"])}/{self.base_len_regions_bottom} notifications in bottom layer',  
        #     ha='center',
        #     fontsize=10
        #     )
        
        

        
        # self.save_im_button_ax = self.fig.add_axes([0.83, 0.15, 0.1, 0.05])
        # self.save_im_button = Button(self.save_im_button_ax, 'Save image and\nannotations')
        # self.save_im_button.on_clicked(self.save_im)
        

        self.save_button_ax = self.fig.add_axes([0.83, 0.05, 0.1, 0.05])
        self.save_button = Button(self.save_button_ax, 'Save state, paths and\n calculate snow depth')
              
        self.pending_save = False
        self.save_button.on_clicked(self.handle_save_click)
        
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_keypress = self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.cid_keyrelease = self.fig.canvas.mpl_connect('key_release_event', self.on_keyrelease)

        self.radio.on_clicked(self.on_layer_change)
        self.button.on_clicked(self.on_rerun)
        self.reset_button.on_clicked(self.on_reset)
        self.toggle_button.on_clicked(self.toggle_paths)
        self.quickboost_toggle.on_clicked(self.toggle_quickboost)
        self.internal_toggle_button.on_clicked(self.toggle_internal_layers)

        # self.next_button.on_clicked(self.next_chunk)
        # self.prev_button.on_clicked(self.prev_chunk)
        if self.has_flightstate:
            self.add_var_panel()
        self.add_overview_panel()
        self.set_chunk_limits()
        self.update_points()

        plt.show()
        
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()
        
    # def save_im(self, event):
    #     chunk_data = self.section[:, self.chunk_start:self.chunk_end]
    #     chunk_data_norm = (chunk_data - chunk_data.min()) / (chunk_data.max() - chunk_data.min())
    #     chunk_data_uint8 = (chunk_data_norm * 255).astype('uint8')

    #     #save image
    #     img = Image.fromarray(chunk_data_uint8)
    #     img.save(f'images/{self.uwibass.dataset_name}_chunk{self.current_chunk}_image.png', compress_level=0)  # PNG is lossless
        
    #     #save annotation
    #     annotation = np.zeros_like(chunk_data, dtype=np.uint8)
    #     smooth_top = pd.Series(self.path_top[self.chunk_start:self.chunk_end]).rolling(window=5, center=True, min_periods=0).mean().astype('int').values
    #     smooth_bottom = pd.Series(self.path_bottom[self.chunk_start:self.chunk_end]).rolling(window=5, center=True, min_periods=0).mean().astype('int').values
        
    #     for i in range(chunk_data.shape[1]):
    #         annotation[smooth_top[i]:smooth_bottom[i], i] = 255
    #     annotation_img = Image.fromarray(annotation)
    #     annotation_img.save(f'annotations/{self.uwibass.dataset_name}_chunk{self.current_chunk}_annotation.png', compress_level=0)  # PNG is lossless
        
    
    def toggle_internal_layers(self, event):
        self.show_internal = not self.show_internal
        self.update_internal()

    def update_internal(self):
        for artist in self.internal_artists:
            artist.remove()
        self.internal_artists.clear()
        
        if self.show_internal:
            for i, p in enumerate(self.internal_paths):
                py, px = list(zip(*p))
                py = np.array(py)
                px = np.array(px)
                px_tmp = px.copy()
                px = px[(px_tmp >= self.chunk_start) & (px_tmp < self.chunk_end)]
                py = py[(px_tmp >= self.chunk_start) & (px_tmp < self.chunk_end)]
                if len(px) > 0:
                    internal_artist = self.ax.plot(px, py + self.path_top[px] , 'k-', lw=1, alpha=0.5,
                                                   color=(188/255, 232/255, 67/255)
                                                #    color = self.cmap_internal(self.norm_internal(self.internal_SNR[i]))
                                                   )[0]
                    self.internal_artists.append(internal_artist)
        self.fig.canvas.draw_idle()

        
    def handle_save_click(self, event):
        if self.pending_save:
            self._save_to_RADAR()
            self.pending_save = False
        else:
            self.on_save_paths(event)
            
            
    def on_save_paths(self, event):
        if hasattr(self.RADAR, 'PF_top_interface') or hasattr(self.RADAR, 'PF_bottom_interface'):
            self.status_text.set_text("Click again\nto overwrite.")
            self.pending_save = True
            self.fig.canvas.draw_idle()
            self.fig.canvas.new_timer(interval=1, callbacks=[(self.reset_pending_save, [], {})]).start()
            return

        self._save_to_RADAR()

    def reset_pending_save(self):
        self.pending_save = False
        self.status_text.set_text("")
        self.fig.canvas.draw_idle()

    def _save_to_RADAR(self):
        self.RADAR.PF_cost_maps = {}
        self.RADAR.PF_cost_maps['top'] = self.cost_top
        self.RADAR.PF_cost_maps['bottom'] = self.cost_bottom
        self.RADAR.PF_cost_maps['unbiased'] = self.unbiased_cost
        
        self.RADAR.PF_top_interface = np.array(self.path_top)
        self.RADAR.PF_bottom_interface = np.array(self.path_bottom)

        range_air = self.RADAR.range_air
        
        # TODO is this really necessary? do we ever have multiple snow pits at one site?
        range_snow = np.column_stack([range_air / np.sqrt(eps_r) for eps_r in self.RADAR.SP_bulk_eps_r])

        self.RADAR.range_snow = range_snow
        self.RADAR.PF_snow_depth_air = np.array([range_air[self.RADAR.PF_bottom_interface[i]] - range_air[self.RADAR.PF_top_interface[i]] for i in range(len(self.RADAR.PF_bottom_interface))]) / 100
        self.RADAR.PF_snow_depth = np.array([range_snow[self.RADAR.PF_bottom_interface[i], i] - range_snow[self.RADAR.PF_top_interface[i], i] for i in range(len(self.RADAR.PF_bottom_interface))]) / 100

        self.RADAR.PF_top_interface_elev = [self.RADAR.range_air[p]/100 + self.RADAR.range_air[-1]/100 * (self.RADAR.CTUN_SAlt_filtered[p] // (self.RADAR.range_air[-1]/100)) for p in self.RADAR.PF_top_interface]
        self.RADAR.PF_bottom_interface_elev = self.RADAR.PF_top_interface_elev + self.RADAR.PF_snow_depth

        self.RADAR.PF_radar_uncertainty, self.RADAR.PF_sampling_uncertainty, self.RADAR.PF_top_interface_SNR, self.RADAR.PF_bottom_interface_SNR, self.RADAR.range_resolution_air, self.RADAR.range_resolution_snow = calculate_radar_uncertainties(self.RADAR, self.section, self.unbiased_cost)
        self.RADAR.PF_snow_uncertainty = calculate_snow_depth_uncertainty(self.RADAR)
        self.RADAR.PF_total_uncertainty = calculate_total_uncertainty(self.RADAR)


        # self.uwibass.PF_top_interface_roughness = calculate_topo_roughness(self.uwibass.log_UTM_x, self.uwibass.log_UTM_y, self.uwibass.PF_top_interface_elev,
        #                                  lengthscale=2)
        # self.uwibass.PF_bottom_interface_roughness = calculate_topo_roughness(self.uwibass.log_UTM_x, self.uwibass.log_UTM_y, self.uwibass.PF_bottom_interface_elev,
        #                                  lengthscale=2)
        
        # TODO this ddoes not yet work for non-uwibass datasets, need a solution for the path variable in RADAR
        with open(self.full_path, 'wb') as outp:
            pickle.dump(self.RADAR, outp, pickle.HIGHEST_PROTOCOL)
        
        # TODO save the important (e.g. non-2D) data to a .csv?
        # uwibass_df = pd.DataFrame({
        #     'timestamp': self.uwibass.datetime_timestamp,
        #     'x': self.uwibass.log_UTM_x,
        #     'y': self.uwibass.log_UTM_y,
        #     'lon': self.uwibass.GPS_Lng,
        #     'lat': self.uwibass.GPS_Lat,
        #     'alt': self.uwibass.GPS_Alt,
        #     'rngf': self.uwibass.CTUN_SAlt,
        #     'roll': self.uwibass.roll,
        #     'pitch': self.uwibass.pitch,
        # })
        self.status_text.set_text("Paths saved to RADAR.")
        self.fig.canvas.draw_idle()


    def add_overview_panel(self):
        self.ax_overview.imshow(Quickboost(self.section,.5)[::5,::5], cmap=cmr.neutral,
                                interpolation='none',
                                aspect=.5 if self.section.shape[1] < 20000 else 1,
                                extent=[0, self.section.shape[1], self.section.shape[0], 0])
            
        self.line_top_overview, = self.ax_overview.plot(range(len(self.path_top)),self.path_top , color='deepskyblue', lw=2, alpha=0.5, zorder=2)
        self.line_bottom_overview, = self.ax_overview.plot(range(len(self.path_bottom)), self.path_bottom, color='magenta', lw=2, alpha=0.5, zorder=2)
        
        if self.add_grounddata:
        #     x_vals_MP = np.arange(len(self.path_top))
        #     y_vals_MP = self.path_top + np.where(self.uwibass.MP_snow_depth_in_indices.astype(int) != 0, self.uwibass.MP_snow_depth_in_indices.astype(int), np.nan)
        #     self.mp_scatter_overview.set_offsets(np.column_stack([x_vals_MP, y_vals_MP]))

            self.update_insitu()

        self.draw_chunk_box()
        # self.draw_regions()
        self.fig.canvas.mpl_connect('button_press_event', self.on_overview_click)

    
    def add_var_panel(self):
        self.var1.set_data(range(len(self.RADAR.roll)), self.RADAR.roll)
        self.var2.set_data(range(len(self.RADAR.pitch)), self.RADAR.compensated_pitch if hasattr(self.RADAR, 'compensated_pitch') else self.RADAR.pitch)
        self.var3.set_data(range(len(self.RADAR.altitude_mask)), self.RADAR.altitude_mask)
        self.var1.set_label('Roll')
        self.var2.set_label('Compensated Pitch' if hasattr(self.RADAR, 'compensated_pitch') else 'Pitch')

        self.ax_var.legend(loc='center left', bbox_to_anchor=(0, 1.2), fontsize=10, ncols=3, frameon=False)
        self.draw_chunk_box()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_overview_click)
        
    # def decimate_regions_by_counter(self):
                
    #     for layer in ['top', 'bottom']:
    #         new_regions = []
            
    #         for reg in self.regions[layer]:
    #             recompute_median = np.median(self.recompute_counter[layer][reg['start']:reg['end']])
                
    #             if recompute_median < 2:
    #                 new_regions.append(reg)
                
    #         self.regions[layer] = new_regions
            
    
    # def draw_regions(self):
        
    #     for patch in getattr(self, 'overview_region_vspans', []):
    #         patch.remove()
            
    #     for patch in getattr(self, 'ax_region_vspans', []):
    #         patch.remove()
                
    #     self.overview_region_vspans = []
    #     self.ax_region_vspans = []

    #     # if self.active_layer == 'top':
    #     cmap = plt.cm.Greens
    #     norm = plt.Normalize(vmin=0, vmax=max(reg['combined_score'] for reg in self.regions['top']))
    #     for reg in self.regions['top']:
    #         vspan = self.ax_overview.axvspan(reg['start'], reg['end'], color=cmap(norm(reg['combined_score'])), alpha=0.2, lw=0, zorder=1)
    #         self.overview_region_vspans.append(vspan)
            
    #         if self.chunk_start <= reg['start'] < self.chunk_end or self.chunk_start < reg['end'] <= self.chunk_end:
    #             a = np.min(self.path_top[reg['start']:reg['end']]) - 5
    #             b = np.max(self.path_top[reg['start']:reg['end']]) + 5
    #             if a>b : a, b = b, a
    #             ymin, ymax = self.ax.get_ylim()
    #             dy = ymax-ymin
    #             lims = ((y-ymin)/dy for y in (a, b))
    #             vspan = self.ax.axvspan(reg['start'],
    #                                     reg['end'],
    #                                     *lims,
    #                                     color=cmap(norm(reg['combined_score'])), alpha=0.2, lw=0, zorder=1)
    #             self.ax_region_vspans.append(vspan)

    #     # elif self.active_layer == 'bottom':
    #     cmap = plt.cm.Oranges
    #     norm = plt.Normalize(vmin=0, vmax=max(reg['combined_score'] for reg in self.regions['bottom']))
    #     for reg in self.regions['bottom']:
    #         vspan = self.ax_overview.axvspan(reg['start'], reg['end'], color=cmap(norm(reg['combined_score'])), alpha=0.2, lw=0, zorder=1)
    #         self.overview_region_vspans.append(vspan)
            
    #         if self.chunk_start <= reg['start'] < self.chunk_end or self.chunk_start < reg['end'] <= self.chunk_end:
    #             a = np.min(self.path_bottom[reg['start']:reg['end']]) - 5
    #             b = np.max(self.path_bottom[reg['start']:reg['end']]) + 5
    #             if a>b : a, b = b, a
    #             ymin, ymax = self.ax.get_ylim()
    #             dy = ymax-ymin
    #             lims = ((y-ymin)/dy for y in (a, b))
    #             vspan = self.ax.axvspan(reg['start'],
    #                                     reg['end'],
    #                                     *lims,
    #                                     color=cmap(norm(reg['combined_score'])), alpha=0.2, lw=0, zorder=1)
    #             self.ax_region_vspans.append(vspan)
                
            
    def draw_chunk_box(self):
        # Remove previous chunk box patches
        for patch in getattr(self, 'chunk_box_patches', []):
            patch.remove()
        self.chunk_box_patches = []

        # Rectangle in var
        if self.has_flightstate:
            rect_var = Rectangle(
                (self.chunk_start, np.min([self.var1.get_ydata(), self.var2.get_ydata()])),
                self.chunk_end - self.chunk_start,
                np.max([self.var1.get_ydata(), self.var2.get_ydata()]) + np.abs(np.min([self.var1.get_ydata(), self.var2.get_ydata()])),
                linewidth=1, edgecolor='black', facecolor='none', alpha=0.7, zorder=10
            )
            self.ax_var.add_patch(rect_var)
            self.chunk_box_patches.append(rect_var)
        
        # Rectangle in overview
        rect_overview = Rectangle(
            (self.chunk_start, 0),
            self.chunk_end - self.chunk_start,
            self.section.shape[0],
            linewidth=1, edgecolor='black', facecolor='none', alpha=0.7, zorder=10
        )
        self.ax_overview.add_patch(rect_overview)
        self.chunk_box_patches.append(rect_overview)

        # Rectangle in zoomed ax
        rect_ax = Rectangle(
            (self.chunk_start, 0),
            self.chunk_end - self.chunk_start,
            self.section.shape[0],
            linewidth=1, edgecolor='black', facecolor='none', alpha=0.7, zorder=10
        )
        self.ax.add_patch(rect_ax)
        self.chunk_box_patches.append(rect_ax)

        # Draw lines from lower corners in overview to upper corners in ax
        lower_corners_overview = [
            (self.chunk_start, self.section.shape[0]),
            (self.chunk_end, self.section.shape[0])
        ]
        upper_corners_ax = [
            (self.chunk_start, 0),
            (self.chunk_end, 0)
        ]
        for xyA, xyB in zip(lower_corners_overview, upper_corners_ax):
            con = ConnectionPatch(
                xyA=xyA, coordsA=self.ax_overview.transData,
                xyB=xyB, coordsB=self.ax.transData,
                color='black', linewidth=0.7, alpha=0.5, zorder=11
            )
            self.fig.add_artist(con)
            self.chunk_box_patches.append(con)


    def on_overview_click(self, event):
        if event.inaxes != self.ax_overview:
            return
        # new_chunk = int(event.xdata // self.chunk_size)
        step_size = self.chunk_size - self.overlap
        new_chunk = int(event.xdata // step_size)

        if 0 <= new_chunk < self.num_chunks:
            self.current_chunk = new_chunk
            self.set_chunk_limits()

         
    def toggle_insitu(self, event):
        self.show_insitu = not self.show_insitu
        self.update_insitu()
        
    def update_insitu(self):
        MP_x_vals = np.arange(len(self.path_top))
        MP_y_vals = self.path_top + np.where(self.RADAR.MP_snow_depth_in_range_bins.astype(int) != 0, self.RADAR.MP_snow_depth_in_range_bins.astype(int), np.nan)
        # MP_y_vals = self.path_top + np.where(self.uwibass.MP_snow_depth_in_range_bins.astype(int) != 0, -20, np.nan)
        
        self.mp_scatter_overview.set_offsets(np.column_stack([MP_x_vals, np.where(self.RADAR.MP_snow_depth_in_range_bins.astype(int) != 0, -100, np.nan)]))
        
        if self.df_SP is not None:
            meshed_indices, meshed_top_layers = np.meshgrid(self.RADAR.SP_indices , self.RADAR.SP_top_layers)
            _, meshed_bottom_layers = np.meshgrid(self.RADAR.SP_indices, self.RADAR.SP_bottom_layers)

            meshed_top_layers = meshed_top_layers + self.path_top[self.RADAR.SP_indices]
            meshed_bottom_layers = meshed_bottom_layers + self.path_top[self.RADAR.SP_indices]

            self.snow_profile_overview.set_offsets(np.column_stack([self.RADAR.SP_indices, [-100] * len(self.RADAR.SP_indices)]))

        if self.show_insitu == True:
            self.mp_scatter.set_offsets(np.column_stack([MP_x_vals[self.chunk_start:self.chunk_end], MP_y_vals[self.chunk_start:self.chunk_end]]))
            if self.df_SP is not None:
                self.snow_profile.set_offsets(np.column_stack([np.concatenate((meshed_indices.flatten(), meshed_indices.flatten())), np.concatenate((meshed_top_layers.flatten(), meshed_bottom_layers.flatten()))]))
        else:
            self.mp_scatter.set_offsets(np.empty((0, 2)))
            if self.df_SP is not None:

                self.snow_profile.set_offsets(np.empty((0, 2)))

        self.fig.canvas.draw_idle()

    def set_chunk_limits(self):
        self.chunk_start, self.chunk_end = self.get_chunk_bounds()

        chunk_data = self.section[:, self.chunk_start:self.chunk_end]
        if self.use_quickboost:
            chunk_data = Quickboost(chunk_data, degree=0.1)

        self.im.set_data(chunk_data)
        self.im.set_extent((self.chunk_start, self.chunk_end, self.section.shape[0], 0))
        self.im.set_clim(chunk_data.min(), chunk_data.max())

        self.line_top.set_data(range(self.chunk_start, self.chunk_end), self.path_top[self.chunk_start:self.chunk_end])
        self.line_bottom.set_data(range(self.chunk_start, self.chunk_end), self.path_bottom[self.chunk_start:self.chunk_end])
        
        self.line_top_overview.set_data(range(len(self.path_top)), self.path_top)
        self.line_bottom_overview.set_data(range(len(self.path_bottom)), self.path_bottom)
        if self.add_grounddata:
            self.update_insitu()
        if self.show_internal:
            self.update_internal()

        # self.chunk_text.set_text(f"Chunk {self.current_chunk + 1} / {self.num_chunks}")

        # active_cost = self.cost_top if self.active_layer == 'top' else self.cost_bottom
        # self.cost_im.set_data(active_cost[:, self.chunk_start:self.chunk_end])
        # self.cost_im.set_extent((self.chunk_start, self.chunk_end, active_cost.shape[0], 0))
        self.draw_chunk_box()
        # self.draw_regions()
        if self.has_geolocation:
            self.map_chunk.set_data(self.RADAR.UTM_x[self.chunk_start:self.chunk_end], 
                                    self.RADAR.UTM_y[self.chunk_start:self.chunk_end])
        
        # self.map_ax.set_xlim(self.map_xlims[0] - 50, self.map_xlims[1] + 50)
        # self.map_ax.set_ylim(self.map_ylims[0] - 50, self.map_ylims[1] + 50)
        # self.map_ax.plot(self.uwibass.log_UTM_x[self.chunk_start:self.chunk_end], self.uwibass.log_UTM_y[self.chunk_start:self.chunk_end], color='black', lw=5, alpha=1, zorder=0)

        self.ax.set_xlim(self.chunk_start, self.chunk_end)
        self.ax.set_ylim(self.section.shape[0], 0)
        self.fig.canvas.draw_idle()

    def on_layer_change(self, label):
        self.active_layer = label
        # self.set_chunk_limits()
        # self.draw_regions()
        self.update_points()

    def update_cost_maps_after_path_change(self):
        self.cost_top = self.base_cost_top.copy()
        self.cost_bottom = self.base_cost_bottom.copy()
        
        diff = self.path_bottom - self.path_top
        
        masking_strengths = np.zeros(len(diff))
        masking_strengths[diff > 0] = 1
        if self.active_layer == 'both':
            for start, end in self.merged_windows:
                masking_strengths[start:end] = 0
                
        self.cost_top = mask_path_in_cost2(self.cost_top, zip(self.path_bottom, range(len(self.path_bottom))), radius=3, strength=masking_strengths)
        self.cost_bottom = mask_path_in_cost2(self.cost_bottom, zip(self.path_top, range(len(self.path_top))), radius=3, strength=masking_strengths)
        
        # Also re-apply user clicks to the cost maps
        for (x, y) in self.all_clicked_points['top']:
            if 0 <= y < self.cost_top.shape[0] and 0 <= x < self.cost_top.shape[1]:
                self.cost_top[y-1:y+1, x-3:x+3] = -10
                
        for (x, y) in self.all_clicked_points['bottom']:
            if 0 <= y < self.cost_bottom.shape[0] and 0 <= x < self.cost_bottom.shape[1]:
                self.cost_bottom[y-1:y+1, x-3:x+3] = -10
                
        self.set_chunk_limits()

    def on_keypress(self, event):
        if event.key == 'enter':
            self.on_rerun(None)
            
        elif event.key == ' ' or event.key == '-':  # spacebar or dash
            self.space_pressed = True
            toolbar = plt.get_current_fig_manager().toolbar
            if toolbar is not None:
                toolbar.pan()
                toolbar._active = None
                
        elif event.key in ('right', '→'):
            self.next_chunk(None)
            
        elif event.key in ('left', '←'):
            self.prev_chunk(None)
        
        elif event.key in ('up', '↑', 'down', '↓'):
            labels = self.layer_labels
            idx = labels.index(self.active_layer)
            if event.key in ('up', '↑'):
                new_idx = (idx - 1) % len(labels)
            else:  # down or ↓
                new_idx = (idx + 1) % len(labels)
            # this will call on_layer_change for you
            self.radio.set_active(new_idx)
            
 
    def on_keyrelease(self, event):
        if event.key == ' ' or event.key ==  '-':  # spacebar or dash
            self.space_pressed = False
            
    def on_click(self, event):
        if not event.inaxes == self.ax:
            return
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar.mode in ['pan/zoom', 'zoom rect'] and not self.space_pressed:
            return
        self.dragging = True
        self.add_point(event)

    def on_motion(self, event):
        if self.dragging and event.inaxes == self.ax:
            self.add_point(event)

    def on_release(self, event):
        self.dragging = False

    def toggle_quickboost(self, label):
        self.use_quickboost = not self.use_quickboost
        self.set_chunk_limits()

    def get_chunk_bounds(self):
        start = max(0, self.current_chunk * (self.chunk_size - self.overlap))
        end = min(self.section.shape[1], start + self.chunk_size)
        return start, end

    def next_chunk(self, event=None):
        if self.current_chunk < self.num_chunks - 1:
            self.current_chunk += 1
            self.set_chunk_limits()            

    def prev_chunk(self, event=None):
        if self.current_chunk > 0:
            self.current_chunk -= 1
            self.set_chunk_limits()

    def add_point(self, event):
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        
        if self.active_layer == 'top':
            cost = self.cost_top
            
        elif self.active_layer == 'bottom':
            cost = self.cost_bottom
        
        if self.active_layer == 'both':
            cost = self.unbiased_cost
            y_range = range(max(0, y - 1), min(cost.shape[0], y + 2))
            local_min_y = min(y_range, key=lambda yy: cost[yy, x])
            if (x, local_min_y) not in self.clicked_points['top']:
                self.clicked_points['top'].append((x, local_min_y))
                self.update_points()
                
            cost = self.unbiased_cost
            local_min_y = min(y_range, key=lambda yy: cost[yy, x])
            if (x, local_min_y) not in self.clicked_points['bottom']:
                self.clicked_points['bottom'].append((x, local_min_y))
                self.update_points()
            
        else:
            y_range = range(max(0, y - 1), min(cost.shape[0], y + 2))
            local_min_y = min(y_range, key=lambda yy: cost[yy, x])
            if (x, local_min_y) not in self.clicked_points[self.active_layer]:
                self.all_clicked_points[self.active_layer] = [point for point in self.all_clicked_points[self.active_layer] if point[0] != x]
                self.clicked_points[self.active_layer] = [point for point in self.clicked_points[self.active_layer] if point[0] != x]
                self.clicked_points[self.active_layer].append((x, local_min_y))
                self.update_points()
                    
    def on_layer_change(self, label):
        self.active_layer = label
        self.update_points()
        # self.draw_regions()

    def on_reset(self, event):
        self.clicked_points = {'top': [], 'bottom': []}
        self.update_points()

    def toggle_paths(self, event):
        self.paths_visible = not self.paths_visible
        self.line_top.set_visible(self.paths_visible)
        self.line_bottom.set_visible(self.paths_visible)
        self.fig.canvas.draw_idle()

    def update_points(self):
        for artist in self.point_artists:
            artist.remove()
        self.point_artists.clear()
        if self.active_layer == 'both':
            for layer in ['top', 'bottom']:
                for (x, y) in self.clicked_points[layer]:
                    if self.chunk_start <= x < self.chunk_end:
                        artist = self.ax.plot(x, y, 'gx')[0]
                
                        self.point_artists.append(artist)
        else:
            for (x, y) in self.clicked_points[self.active_layer]:
                if self.chunk_start <= x < self.chunk_end:
                    if self.active_layer == 'top':
                        artist = self.ax.plot(x, y, 'rx')[0]
                    elif self.active_layer == 'bottom':
                        artist = self.ax.plot(x, y, 'bx')[0]
            
                    self.point_artists.append(artist)

        self.fig.canvas.draw_idle()
#  elif self.active_layer == 'both':
#                     artist = self.ax.plot(x, y, 'gx')[0]
    def merge_windows(self, points):
        indices = sorted(set(x for x, _ in points))
        if not indices:
            return []
        if self.active_layer != 'both':
            windows = [(max(0, i - self.window_size), min(self.section.shape[1], i + self.window_size)) for i in indices]
            merged = [windows[0]]
            for start, end in windows[1:]:
                prev_start, prev_end = merged[-1]
                if start <= prev_end:
                    merged[-1] = (prev_start, max(prev_end, end))
                else:
                    merged.append((start, end))
                    
        else:
            if len(indices) == 1:
                merged = [(indices[0] - self.window_size, indices[0] + self.window_size)]
            elif len(indices) == 2:
                merged = [(min(indices), max(indices))]
            elif len(indices) > 2:
                windows = [(max(0, i - self.window_size), min(self.section.shape[1], i + self.window_size)) for i in indices]
                merged = [windows[0]]
                for start, end in windows[1:]:
                    prev_start, prev_end = merged[-1]
                    if start <= prev_end:
                        merged[-1] = (prev_start, max(prev_end, end))
                    else:
                        merged.append((start, end))
        return merged

    def on_rerun(self, event):
        
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        self.button_ax.set_visible(False)
        self.fig.canvas.draw_idle()

        for layer, cost in zip(['top', 'bottom'], [self.cost_top, self.cost_bottom]):
            path = self.path_top if layer == 'top' else self.path_bottom
            
            self.merged_windows = self.merge_windows(self.clicked_points[layer])
            
            for start, end in self.merged_windows:
                
                for x, y in self.clicked_points[layer]:
                    
                    self.all_clicked_points[layer].append((x, y))
                    
                    if start <= x < end and 0 <= y < cost.shape[0]:
                        cost[y-1:y+1, x-1:x+1] -= 10
                        
                sub_cost = cost[:, start:end].copy()
                
                y_start = path[start] if start < len(path) else path[-1]
                y_end = path[end - 1] if (end - 1) < len(path) else path[-1]
                sub_cost[y_start, 0] -= 1
                sub_cost[y_end, -1] -= 1

                #TODO: before, this was find_optimal_path2, but I am not sure anymore what exactly the difference is
                refined_path, _ = find_optimal_path(sub_cost, self.RADAR.PF_parameters, layer=layer)
                refined_y = np.array([y for y, _ in refined_path])

                if layer == 'top':
                    self.path_top[start:end] = refined_y
                    
                elif layer == 'bottom':
                    self.path_bottom[start:end] = refined_y

                self.recompute_counter[layer][start:end] += 1

        self.update_cost_maps_after_path_change()
        self.set_chunk_limits()

        # self.regions['top'] = find_flag_regions(self.section, self.path_top)
        # self.regions['bottom'] = find_flag_regions(self.section, self.path_bottom)
        
        # self.decimate_regions_by_counter()
        # self.draw_regions()
        # self.top_regions_text.set_text(f'{len(self.regions["top"])}/{self.base_len_regions_top} notifications in top layer')
        # self.bottom_regions_text.set_text(f'{len(self.regions["bottom"])}/{self.base_len_regions_bottom} notifications in bottom layer')
        
        self.ax.set_xlim(current_xlim)
        self.ax.set_ylim(current_ylim)

        self.clicked_points = {'top': [], 'bottom': []}
        self.update_points()
        self.button_ax.set_visible(True)
        self.fig.canvas.draw_idle()

