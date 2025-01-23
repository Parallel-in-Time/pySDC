def plot_williamson_5(
    results_file_name='./results/williamson_5/field_output.nc', plot_stem='williamson_5'
):  # pragma: no cover
    """
    Plot the results of the Williamson 5 test case from tutorial step 7/F.
    This file is taken from the Gusto example scripts and stores plots in the current directory.

    See https://github.com/tommbendall/tomplot for obtaining tomplot

    The initial conditions are plotted:
    (a) the velocity field, (b) the depth field.

    And at the last recorded time, this plots:
    (a) the relative vorticity field, (b) free-surface height.

    Args:
        results_file_name (str): Path to netCDF results file from Williamson 5 test case
        plot_stem (str): Name of the resulting plots
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import numpy as np
    from netCDF4 import Dataset
    from tomplot import (
        set_tomplot_style,
        tomplot_cmap,
        plot_contoured_field,
        add_colorbar_ax,
        plot_field_quivers,
        tomplot_field_title,
        extract_gusto_coords,
        extract_gusto_field,
        regrid_horizontal_slice,
    )

    # ---------------------------------------------------------------------------- #
    # Initial plot details
    # ---------------------------------------------------------------------------- #
    init_field_names = ['u', 'D']
    init_colour_schemes = ['Oranges', 'YlGnBu']
    init_field_labels = [r'$|u|$ (m s$^{-1}$)', r'$D$ (m)']
    init_contours_to_remove = [None, None, None]
    init_contours = [np.linspace(0, 20, 9), np.linspace(3800, 6000, 12)]

    # ---------------------------------------------------------------------------- #
    # Final plot details
    # ---------------------------------------------------------------------------- #
    final_field_names = ['RelativeVorticity', 'D_plus_topography']
    final_colour_schemes = ['RdBu_r', 'PiYG']
    final_field_labels = [r'$\zeta \ / $ s$^{-1}$', r'$D+B$ (m)']
    final_contours = [np.linspace(-1e-4, 1e-4, 21), np.linspace(5000, 6000, 11)]
    final_contours_to_remove = [0.0, None]

    # ---------------------------------------------------------------------------- #
    # General options
    # ---------------------------------------------------------------------------- #
    projection = ccrs.Robinson()
    contour_method = 'contour'
    # xlims = [-180, 180]
    # ylims = [-90, 90]

    cbar_format = {'RelativeVorticity': '1.1e', 'u': '1.0f', 'D': '1.0f', 'D_plus_topography': '1.0f'}

    # We need to regrid onto lon-lat grid -- specify that here
    lon_1d = np.linspace(-180.0, 180.0, 120)
    lat_1d = np.linspace(-90, 90, 120)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='ij')

    # Things that are likely the same for all plots --------------------------------
    set_tomplot_style()
    data_file = Dataset(results_file_name, 'r')

    # ---------------------------------------------------------------------------- #
    # INITIAL PLOTTING
    # ---------------------------------------------------------------------------- #
    fig = plt.figure(figsize=(15, 5))
    time_idx = 0

    for i, (field_name, colour_scheme, field_label, contour_to_remove, contours) in enumerate(
        zip(init_field_names, init_colour_schemes, init_field_labels, init_contours_to_remove, init_contours)
    ):

        # Make axes
        ax = fig.add_subplot(1, 2, 1 + i, projection=projection)

        # Data extraction ----------------------------------------------------------
        if field_name == 'u':
            zonal_data = extract_gusto_field(data_file, 'u_zonal', time_idx=time_idx)
            meridional_data = extract_gusto_field(data_file, 'u_meridional', time_idx=time_idx)
            field_data = np.sqrt(zonal_data**2 + meridional_data**2)
            coords_X, coords_Y = extract_gusto_coords(data_file, 'u_zonal')

        else:
            field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
            coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
        time = data_file['time'][time_idx] / (24.0 * 60.0 * 60.0)

        # Regrid onto lon-lat grid
        field_data = regrid_horizontal_slice(lon_2d, lat_2d, coords_X, coords_Y, field_data, periodic_fix='sphere')

        # Plot data ----------------------------------------------------------------
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=contour_to_remove)
        cf, _ = plot_contoured_field(
            ax,
            lon_2d,
            lat_2d,
            field_data,
            contour_method,
            contours,
            cmap=cmap,
            line_contours=lines,
            projection=projection,
        )

        add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10, cbar_format=cbar_format[field_name])
        tomplot_field_title(ax, None, minmax=True, field_data=field_data)

        # Add quivers --------------------------------------------------------------
        if field_name == 'u':
            # Need to re-grid to lat-lon grid to get sensible looking quivers
            regrid_zonal_data = regrid_horizontal_slice(
                lon_2d, lat_2d, coords_X, coords_Y, zonal_data, periodic_fix='sphere'
            )
            regrid_meridional_data = regrid_horizontal_slice(
                lon_2d, lat_2d, coords_X, coords_Y, meridional_data, periodic_fix='sphere'
            )
            plot_field_quivers(
                ax,
                lon_2d,
                lat_2d,
                regrid_zonal_data,
                regrid_meridional_data,
                spatial_filter_step=6,
                magnitude_filter=1.0,
                projection=ccrs.PlateCarree(),
            )

    # Save figure ------------------------------------------------------------------
    fig.subplots_adjust(wspace=0.25)
    plt.suptitle(f't = {time:.1f} days')
    plot_name = f'{plot_stem}_initial.png'
    print(f'Saving figure to {plot_name}')
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------------------------- #
    # FINAL PLOTTING
    # ---------------------------------------------------------------------------- #
    fig = plt.figure(figsize=(15, 5))
    time_idx = -1

    for i, (field_name, colour_scheme, field_label, contours, contour_to_remove) in enumerate(
        zip(final_field_names, final_colour_schemes, final_field_labels, final_contours, final_contours_to_remove)
    ):

        # Make axes
        ax = fig.add_subplot(1, 2, 1 + i, projection=projection)

        # Data extraction ----------------------------------------------------------
        field_data = extract_gusto_field(data_file, field_name, time_idx=time_idx)
        coords_X, coords_Y = extract_gusto_coords(data_file, field_name)
        time = data_file['time'][time_idx] / (24.0 * 60.0 * 60.0)

        # Regrid onto lon-lat grid
        field_data = regrid_horizontal_slice(lon_2d, lat_2d, coords_X, coords_Y, field_data, periodic_fix='sphere')

        # Plot data ----------------------------------------------------------------
        cmap, lines = tomplot_cmap(contours, colour_scheme, remove_contour=contour_to_remove)
        cf, _ = plot_contoured_field(
            ax,
            lon_2d,
            lat_2d,
            field_data,
            contour_method,
            contours,
            cmap=cmap,
            line_contours=lines,
            projection=projection,
        )

        add_colorbar_ax(ax, cf, field_label, location='bottom', cbar_labelpad=-10, cbar_format=cbar_format[field_name])
        tomplot_field_title(ax, None, minmax=True, field_data=field_data)

    # Save figure ------------------------------------------------------------------
    plt.suptitle(f't = {time:.1f} days')
    plot_name = f'{plot_stem}_final.png'
    print(f'Saving figure to {plot_name}')
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--results_file_name',
        help='Path to netCDF result file from the Williamson 5 test case',
        type=str,
        default='./results/williamson_5/field_output.nc',
    )
    parser.add_argument('--plot_stem', help='Name of the plots', type=str, default='williamson_5')
    args, unknown = parser.parse_known_args()

    plot_williamson_5(**vars(args))
