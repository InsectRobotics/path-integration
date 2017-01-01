import os
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

nature_single = 89.0 / 25.4
nature_double = 183.0 / 25.4
nature_full = 247.0 / 25.4

mpl.rc('font', family='Arial', size=14)

tl2_color = '#835EAC'
cl1_color = '#B5D770'
compass_color = '#006836'
mem_color = '#DE6F4A'
mem_color_L = '#F37D4D'
mem_color_R = '#FABD5E'
motor_color = '#0071BB'
motor_color_L = '#0071BB'
motor_color_R = '#505AA5'
#flow_color_L = '#806641'
#flow_color_R = '#AE956D'
flow_color_L = 'blue'
flow_color_R = 'green'

PLOT_PATH = 'plots'


def save_plot(fig, filename):
    fig.savefig(os.path.join(PLOT_PATH, filename + '.pdf'),
                bbox_inches='tight', dpi=300)
    #fig.savefig(os.path.join(PLOT_PATH, filename + '.svg'),
    #            bbox_inches='tight', dpi=300)
    #fig.savefig(os.path.join(PLOT_PATH, filename + '.png'),
    #            bbox_inches='tight', dpi=300)


def plot_route(h, v, T_outbound, T_inbound, plot_speed=False,
               plot_heading=False, memory_estimate=None, ax=None, legend=True,
               labels=True, outbound_color='purple', inbound_color='green',
               memory_color='darkorange', quiver_color='gray', title=None,
               label_font_size=11, unit_font_size=10,
               figsize=(nature_single, nature_single)):
    """Plots a route with optional colouring by speed and arrows indicating
    direction."""

    xy = np.vstack([np.array([0.0, 0.0]), np.cumsum(v, axis=0)])
    x, y = xy[:, 0], xy[:, 1]

    lw = 0.5  # Linewidth
    T = T_outbound + T_inbound

    # Generate new plot if no axes passed in.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # Outbound path
    if plot_speed:
        speed = np.clip(np.linalg.norm(np.vstack([np.diff(x), np.diff(y)]),
                        axis=0), 0, 1)
        n_min = np.argmin(speed[:T_outbound])
        n_max = np.argmax(speed[:T_outbound])

        for i in range(T_outbound-1):
            ax.plot(x[i:i+2], y[i:i+2], color=(speed[i], 0.2, 1-speed[i]),
                    lw=lw)

        blue_line = mlines.Line2D([], [], color='blue',
                                  label='Outbound (slow)')
        red_line = mlines.Line2D([], [], color='red', label='Outbound (fast)')
        handles = [blue_line, red_line]
    else:
        line_out, = ax.plot(x[0:T_outbound+1], y[0:T_outbound+1], lw=lw,
                            color=outbound_color, label='Outbound')
        handles = [line_out]

    if plot_heading:
        #interval = T/200  # Good for actual route plots thing (with headwidth 0)
        interval = 20  # Good for memory plot thing (with headwidth 4)
        ax.quiver(x[1:T_outbound:interval], y[1:T_outbound:interval],
                  np.sin(h[1:T_outbound:interval]),
                  np.cos(h[1:T_outbound:interval]),
                  pivot='tail', width=0.003, scale=12.0, headwidth=4, color=quiver_color)
                  #pivot='tail', width=0.002, scale=12.0, color=quiver_color)

    # Inbound path
    line_in, = ax.plot(x[T_outbound:T], y[T_outbound:T], color=inbound_color,
                       lw=lw, label='Return')
    handles.append(line_in)

    # Memory
    if memory_estimate:
        point_estimate = ax.scatter(memory_estimate[0], memory_estimate[1],
                                    color=memory_color, label='Memory')
        handles.append(point_estimate)

    # Nest label
    ax.text(0, 0, 'N', fontsize=12, fontweight='heavy', color='k', ha='center',
            va='center')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=unit_font_size)

    if title:
        ax.set_title(title)

    if labels:
        ax.set_xlabel('Distance (steps)', fontsize=label_font_size)
        ax.set_ylabel('Distance (steps)', fontsize=label_font_size)

    # Legend
    if legend:
        l = ax.legend(handles=handles,
                      loc='best',
                      fontsize=unit_font_size,
                      handlelength=0,
                      handletextpad=0)
        if plot_speed:
            colors = ['blue', 'red', inbound_color]
        else:
            colors = [outbound_color, inbound_color]
        if memory_estimate:
            colors.append(memory_color)
        for i, text in enumerate(l.get_texts()):
            text.set_color(colors[i])
        for handle in l.legendHandles:
            handle.set_visible(False)
        l.draw_frame(False)
    return fig, ax


def plot_motor_trace(ax, motor, T_outbound, T_inbound, outbound_color,
                     return_color, alpha=0.2, label_font_size=11,
                     unit_font_size=10, lw=0.5, t_start=0):
    T = T_outbound + T_inbound
    if T_outbound > 0:
        ax.plot(np.arange(t_start, T_outbound+t_start),
                motor[:T_outbound].T,
                color=outbound_color,
                lw=lw,
                label='Outbound',
                alpha=alpha)
    if T_inbound > 0:
        ax.plot(np.arange(T_outbound+t_start, T+t_start),
                motor[T_outbound:].T,
                color=return_color,
                lw=lw,
                label='Return')

    ax.set_ylabel('$\Sigma$ activity', labelpad=-1, fontsize=label_font_size)
    ax.tick_params(labelsize=unit_font_size)
    ax.set_yticks([-2.5, 0, 2.1])
    ax.set_yticklabels([-3, 0, 3], fontsize=unit_font_size)

    dummy_ax = ax.twinx()
    dummy_ax.set_ylim([-3, 3])
    dummy_ax.set_yticks([-2.5, 2.1])
    dummy_ax.set_yticklabels(['L', 'R'],
                             position=(1.02, 0),
                             fontsize=unit_font_size,
                             va='center',
                             ha='center')


def plot_traces(log, include=['TN1', 'TN2', 'CL1', 'TB1', 'CPU4', 'CPU1', 'motor'],
                fig=None, ax=None, colormap='viridis', title_x=-0.15,
                alpha=0.2, outbound_color='purple', return_color='g',
                label_font_size=11, unit_font_size=10, dashes=[1, 2, 1, 2],
                T_almost_home=None, t_start=0):
    """Generate big plot with all traces of model. Warning: takes long time to
    save!!"""
    T, T_outbound, T_inbound = log.T, log.T_outbound, log.T_inbound
    titles = {'TN1': 'TN1 (Speed)', 'TN2': 'TN2 (Speed)', 'TL2': 'TL2',
              'CL1': 'CL1', 'TB1': 'TB1 (Compass)', 'CPU4': 'CPU4 (Memory)',
              'CPU1': 'CPU1 (Steering)', 'motor': 'motor'}
    data = {'TN1': log.tn1, 'TN2': log.tn2, 'TL2': log.tl2, 'CL1': log.cl1,
            'TB1': log.tb1, 'CPU4': log.cpu4, 'CPU1': log.cpu1,
            'motor': log.motor}

    colors = {'TL2': tl2_color, 'CL1': cl1_color}

    # Generate new plot if no axes passed in.
    if ax is None:
        fig, ax = plt.subplots(len(include), 1,
                               figsize=(nature_single, nature_single))

    N_plots = len(include)
    for i, cell_type in enumerate(include):
        ax[i].set_title(titles[cell_type],
                        x=title_x,
                        y=0.3,
                        va='center',
                        ha='right',
                        fontsize=label_font_size,
                        fontweight='heavy')
        ax[i].set_xticklabels([])
        ax[i].tick_params(labelsize=unit_font_size)

        if cell_type in ['TN1', 'TN2']:
            filtered_l = sp.ndimage.filters.gaussian_filter1d(
                    data[cell_type][0], sigma=20)
            filtered_r = sp.ndimage.filters.gaussian_filter1d(
                    data[cell_type][1], sigma=20)
            tn_l_line, = ax[i].plot(filtered_l, color=flow_color_L, label='L');
            tn_r_line, = ax[i].plot(filtered_r, color=flow_color_R, label='R');
            handles = [tn_l_line, tn_r_line]

            ax[i].plot(data[cell_type][0].T, color=flow_color_L, alpha=0.3,
                       lw=0.5);
            ax[i].plot(data[cell_type][1].T, color=flow_color_R, alpha=0.3,
                       lw=0.5);
            ax[i].set_yticks([0.05, 0.9])
            ax[i].set_yticklabels([0, 1])

            # Make a legend but not for both
            if i % 2 == 0:
                l = ax[i].legend(handles=handles,
                                 bbox_to_anchor=(1.15, 1.2),
                                 loc='upper right',
                                 ncol=1,
                                 fontsize=unit_font_size,
                                 handlelength=0,
                                 handletextpad=0)
                colors = [flow_color_L, flow_color_R]
                for i, text in enumerate(l.get_texts()):
                    text.set_color(colors[i])
                for handle in l.legendHandles:
                    handle.set_visible(False)
                l.draw_frame(False)
            # heatmap plots
        # TODO(tomish) create line plots for CXBasic
        elif cell_type in ['TL2', 'CL1'] and data[cell_type].shape[0] == 1:
            ax[i].plot(data[cell_type][0], color=colors[cell_type]);
            ax[i].set_yticks([-np.pi, np.pi])
            ax[i].set_yticklabels([0, 360])
        elif cell_type in ['TL2', 'CL1', 'TB1', 'CPU4', 'CPU1']:
            # Surface plots related to memory generation.
            p = ax[i].pcolormesh(data[cell_type], vmin=0, vmax=1,
                                 cmap=colormap, rasterized=True);
            ax[i].get_xaxis().set_tick_params(direction='out')
            if cell_type == 'TB1':
                ax[i].set_yticks([1, 7])
                ax[i].set_yticklabels([1, 8])
            else:
                ax[i].set_yticks([1, 14])
                ax[i].set_yticklabels([1, 16])

            if cell_type == 'CPU1':
                # We add alpha to the outbound part
                fig.savefig('dummy.jpg')  # This is needed to force draw plot
                p.get_facecolors().reshape(16, -1, 4)[:, :T_outbound, 3] = 0.1
                p.set_edgecolor('none')
            else:
                p.set_edgecolor('face')
        else:
            # Plots related to steering
            plot_motor_trace(ax[i], log.motor, T_outbound, T_inbound,
                             outbound_color, return_color, alpha,
                             label_font_size, unit_font_size, t_start=t_start);

    # Add label half way (ish) down plot
    ax[0].set_ylabel('Activity', fontsize=label_font_size)
    #ax[1].yaxis.set_label_coords(-0.075, 1.1)

    ax[3].set_ylabel('Cell indices', fontsize=label_font_size)
    ax[3].yaxis.set_label_coords(-0.075, 1.1)

    # Add x labels to bottom plot
    ax[N_plots-1].set_xlabel('Time (steps)', fontsize=label_font_size)
    ax[N_plots-1].get_xaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Colorbar
    ax_cb = fig.add_axes([0.92, 0.257, 0.02, 0.410])
    m = cm.ScalarMappable(cmap=colormap)
    m.set_array(np.linspace(0, 1, 100))
    ax_cb.tick_params(labelsize=unit_font_size)
    cb = fig.colorbar(m, ax_cb)
    cb.set_ticks(np.linspace(0,1,6))
    cb.set_label('Firing rate', size=label_font_size)

    # Top spacer
    ax_space = fig.add_axes([0, 0.92, 1, 0.01])
    ax_space.axis('off')

    # Dotted bars
    if T_almost_home is None:
        T_almost_home = T_outbound + 400  # TODO(tomish) Auto generate this
    v_indices = np.array([0, T_outbound, T_almost_home, T])
    transFigure = fig.transFigure.inverted()
    for i, v_idx in enumerate(v_indices):
        y_max = ax[0].get_ylim()[1]
        coord1 = transFigure.transform(ax[0].transData.transform([v_idx,
                                                                  y_max]))
        coord2 = transFigure.transform(ax[5].transData.transform([v_idx, -3]))
        if i == 0 or i == 3:
            lw = 1
            zorder = 0
        else:
            lw = 1
            zorder = 1
        line = mlines.Line2D((coord1[0], coord2[0]),
                             (coord1[1]+0.06, coord2[1]),
                             transform=fig.transFigure, lw=lw, zorder=zorder,
                             c='w', linestyle='dashed')
        line.set_dashes(dashes)
        fig.lines.append(line)
        line = ax[5].axvline(x=v_idx, lw=lw, c='#333333', linestyle='dashed')
        line.set_dashes(dashes)

    # Labels between bars
    label_indices = (v_indices[:3] + v_indices[1:])/2
    labels = ['Outbound', 'Return', 'Search']
    for i, label_idx in enumerate(label_indices):
        y_max = ax[0].get_ylim()[1]
        ax[0].text(label_idx, y_max*1.2, labels[i], fontsize=label_font_size,
                   va='center', ha='center')

    return fig, ax


def plot_angular_distances(noise_levels, angular_distances, bins=18, ax=None,
                           label_font_size=11, log_scale=False, title=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'),
                               figsize=(nature_single, nature_single))

    colors = [cm.viridis(x) for x in np.linspace(0, 1, len(noise_levels))]

    for i in reversed(range(len(noise_levels))):
        plot_angular_distance_histogram(angular_distance=angular_distances[i],
                                        ax=ax, bins=bins, color=colors[i])

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(22)
    ax.set_title(title, y=1.08, fontsize=label_font_size)

    if log_scale:
        ax.set_rscale('log')
        ax.set_rlim(0.0, 10001)  # What determines this?

    plt.tight_layout()
    return fig, ax


def plot_angular_distance_histogram(angular_distance, ax=None, bins=36,
                                    color='b'):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(nature_single, nature_single))

    radii = np.histogram(angular_distance,
                         np.linspace(-np.pi - np.pi / bins,
                                     np.pi + np.pi / bins,
                                     bins + 2,
                                     endpoint=True))[0]
    radii[0] += radii[-1]
    radii = radii[:-1]
    radii = np.roll(radii, bins/2)
    radii = np.append(radii, radii[0])
    # Set all values to have at least a count of 1
    # Need this hack to get the plot fill to work reliably
    radii[radii == 0] = 1
    theta = np.linspace(0, 2 * np.pi, bins+1, endpoint=True)

    ax.plot(theta, radii, color=color, alpha=0.5)
    if color:
        ax.fill_between(theta, 0, radii, alpha=0.2, color=color)
    else:
        ax.fill_between(theta, 0, radii, alpha=0.2)

    return fig, ax


def plot_route_straightness(cum_min_dist, x_count=500, ax=None,
                            label_font_size=11, unit_font_size=10):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(nature_single*1.2, nature_single))

    # TESTING remove this if necessary
    mu = np.nanmean(cum_min_dist, axis=1)
    sigma = np.nanstd(cum_min_dist, axis=1)
    t = np.linspace(0, 2, x_count)

    ax.plot(t, mu, label='Mean path')
    ax.fill_between(t, mu+sigma, mu-sigma, facecolor='blue', alpha=0.5)
    ax.set_ylim(0, 1)
    ax.plot([0, 1], [1, 0], 'r', label='Best possible path')
    ax.set_xlabel('Distance travelled relative to turning point distance',
                  fontsize=label_font_size)
    ax.set_ylabel('Distance from home', fontsize=label_font_size)
    ax.set_title('Tortuosity of homebound route', y=1.05,
                 fontsize=label_font_size)

    vals = ax.get_xticks()
    ax.set_xticklabels(['{:3.0f}%'.format(x*100) for x in vals])

    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
    ax.tick_params(labelsize=unit_font_size)

    ax.axvline(x=1, ymin=0, ymax=mu[250], color='black', linestyle='dotted')

    ax.annotate(s='',
                xy=(1, mu[250]),
                xytext=(1, 1),
                arrowprops=dict(facecolor='black',
                                arrowstyle='<->'))

    ax.text(1.05, mu[250]+(1-mu[250])/2, '$C$', fontsize=14, color='k',
            ha='left', va='center')

    l = ax.legend(loc='best', prop={'size': 12}, handlelength=0,
                  handletextpad=0)
    colors = ['blue', 'red']
    for i, text in enumerate(l.get_texts()):
        text.set_color(colors[i])
        text.set_ha('right')  # ha is alias for horizontalalignment
        text.set_position((103, 0))
    for handle in l.legendHandles:
        handle.set_visible(False)
    l.draw_frame(False)
    plt.tight_layout()
    return fig, ax


def plot_distance_v_noise(min_dists, min_dist_stds, distances, noise_levels,
                          ax=None, label_font_size=11, unit_font_size=10,
                          title=None, x_lim=10000, y_lim=300):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(nature_single, nature_single))

    colors = [cm.viridis(x) for x in np.linspace(0, 1, len(noise_levels))]

    for i in range(len(noise_levels)):
        noise = noise_levels[i]
        mu = min_dists[i]
        sigma = min_dist_stds[i]
        if noise != 'Random':
            ax.semilogx(distances, mu, color=colors[i], label=noise, lw=1);
        else:
            ax.semilogx(distances, mu, color=colors[i], label='Random walk',
                        lw=1);
        ax.fill_between(distances,
                        [m+s for m, s in zip(mu, sigma)],
                        [m-s for m, s in zip(mu, sigma)],
                        facecolor=colors[i], alpha=0.2);

    ax.set_xlim(10, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_title(title, fontsize=label_font_size)
    ax.tick_params(labelsize=unit_font_size)
    ax.set_xlabel('Route length (steps)', fontsize=label_font_size)
    ax.set_ylabel('Distance (steps)', fontsize=label_font_size)

    handles, labels = ax.get_legend_handles_labels()

    l = ax.legend(handles,
                  labels,
                  loc='best',
                  fontsize=label_font_size,
                  handlelength=0,
                  handletextpad=0,
                  title='Noise:')
    l.get_title().set_fontsize(label_font_size)
    for i, text in enumerate(l.get_texts()):
        text.set_color(colors[i])
    for handle in l.legendHandles:
        handle.set_visible(False)
    l.draw_frame(False)
    plt.tight_layout()
    return fig, ax


def plot_angle_of_motion(h, v, ax=None, label_font_size=11, unit_font_size=10):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(nature_single, nature_single))

    angle_rad = (h - np.arctan2(v[:, 0], v[:, 1]) + np.pi) % (2 * np.pi) - np.pi
    angle_deg = np.rad2deg(angle_rad)
    ax.plot(angle_deg)
    ax.axhline(-45, color='r')
    ax.axhline(45, color='r')
    ax.set_title('Heading - Direction of Motion', fontsize=label_font_size)
    ax.tick_params(labelsize=unit_font_size)
    ax.set_xlabel('Route length (steps)', fontsize=label_font_size)
    ax.set_ylabel('Heading offset (degrees)', fontsize=label_font_size)
    return fig, ax


def plot_speed(v, ax=None, label_font_size=11, unit_font_size=10):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(nature_single, nature_single))

    ax.plot(np.sqrt(v[:, 0]**2 + v[:, 1]**2))
    ax.set_title('Speed', fontsize=label_font_size)
    ax.tick_params(labelsize=unit_font_size)
    ax.set_xlabel('Route length (steps)', fontsize=label_font_size)
    ax.set_ylabel('Speed', fontsize=label_font_size)
    return fig, ax


def plot_cxr_weights(cx, label_font_size=11, unit_font_size=10,
                     colormap='viridis'):
    sources = ['TL2', 'CL1', 'TB1', 'TB1', 'TN', 'TB1', 'TB1', 'CPU4', 'CPU4',
               'CPU4', 'Pontin', 'Pontin']
    targets = ['CL1', 'TB1', 'TB1', 'CPU4', 'CPU4', 'CPU1a', 'CPU1b', 'CPU1a',
               'CPU1b', 'Pontin', 'CPU1a', 'CPU1b']
    ticklabels = {'TL2': range(1, 17),
                  'CL1': range(1, 17),
                  'TB1': range(1, 9),
                  'TN': ['L', 'R'],
                  'CPU4': range(1, 17),
                  'Pontin': range(1, 17),
                  'CPU1a': range(2, 16),
                  'CPU1b': range(8, 10)}

    weights = [-np.eye(16), cx.W_CL1_TB1, -cx.W_TB1_TB1,
               -cx.W_TB1_CPU4, cx.W_TN_CPU4, -cx.W_TB1_CPU1a,
               -cx.W_TB1_CPU1b, cx.W_CPU4_CPU1a, cx.W_CPU4_CPU1b,
               cx.W_CPU4_pontin, -cx.W_pontin_CPU1a, -cx.W_pontin_CPU1b]

    fig, ax = plt.subplots(4, 3, figsize=(12, 16))

    for i in range(12):
        cax = ax[i / 3][i % 3]
        p = cax.pcolor(weights[i], cmap=colormap, vmin=-1, vmax=1)
        p.set_edgecolor('face')
        cax.set_aspect('equal')

        cax.set_xticks(np.arange(weights[i].shape[1]) + 0.5)
        cax.set_xticklabels(ticklabels[sources[i]])

        cax.set_yticks(np.arange(weights[i].shape[0]) + 0.5)
        cax.set_yticklabels(ticklabels[targets[i]])

        if i == 1:
            cax.set_title(sources[i] + ' to ' + targets[i], y=1.41)
        else:
            cax.set_title(sources[i] + ' to ' + targets[i])

        cax.set_xlabel(sources[i] + ' cell indices')
        cax.set_ylabel(targets[i] + ' cell indices')
        cax.tick_params(axis=u'both', which=u'both', length=0)

    cbax = fig.add_axes([1.02, 0.05, 0.02, 0.9])
    m = cm.ScalarMappable(cmap=colormap)
    m.set_array(np.linspace(-1, 1, 100))
    cb = fig.colorbar(m, cbax, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label('Connection Strength', labelpad=-50)
    cb.ax.set_yticklabels(['-1.0 (Inhibition)', '-0.5', '0.0', '0.5',
                           '1.0 (Excitation)'])
    plt.tight_layout()
    return fig, ax
