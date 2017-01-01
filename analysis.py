import numpy as np


def get_xy_from_velocity(V):
    XY = np.cumsum(V, axis=1)
    X = XY[:, :, 0]
    Y = XY[:, :, 1]
    return X, Y


def angular_distance(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


# Put some stuff in here to analyse data
def compute_angular_distance(V, T_outbound, num_steps=20):
    """We look at the homebound path and calculate angle relative to turnin
    point."""
    X, Y = get_xy_from_velocity(V)

    nest_angles = np.arctan2(-X[:, T_outbound], -Y[:, T_outbound])

    if V.shape[1] >= T_outbound+num_steps:
        return_angles = np.arctan2(X[:, T_outbound+num_steps] - X[:, T_outbound],
                                   Y[:, T_outbound+num_steps] - Y[:, T_outbound])
        return angular_distance(nest_angles, return_angles)
    else:
        return np.nan


def compute_path_straightness(V, T_outbound):
    X, Y = get_xy_from_velocity(V)
    N = X.shape[0]

    # Distances to the nest at each homebound point
    D = np.sqrt(X[:, T_outbound:]**2 + Y[:, T_outbound:]**2)
    turn_dists = D[:, 0]

    # Get shortest distance so far to nest at each time step
    # We make the y axis equal, by measuring in terms of proportion of
    # route distance.
    cum_min_dist = np.minimum.accumulate(D.T / turn_dists)

    # Get cumulative speed
    cum_speed = np.cumsum(np.sqrt((V[:, T_outbound:, 0]**2 + V[:, T_outbound:, 1]**2)), axis=1)

    # Now we also make the x axis equal in terms of proportion of distance
    # Time is stretched to compensate for longer/shorter routes
    cum_min_dist_norm = []
    for i in np.arange(N):
        t = cum_speed[i]
        xs = np.linspace(0, turn_dists[i]*2, 500, endpoint=False)
        cum_min_dist_norm.append(np.interp(xs,
                                           t,
                                           cum_min_dist[:, i]))
    return np.array(cum_min_dist_norm).T


def compute_tortuosity(cum_min_dist):
    """Computed with tau = L / C."""
    mu = np.nanmean(cum_min_dist, axis=1)
    tortuosity = 1.0 / (1.0 - mu[len(mu)/2])
    return tortuosity


def compute_closest_to_nest(V, T_outbound):
    X, Y = get_xy_from_velocity(V)
    D = np.sqrt(X[:, T_outbound:]**2 + Y[:, T_outbound:]**2)
    min_dists = np.nanmin(D, axis=1)

    d_mu = np.mean(min_dists)
    d_sigma = np.std(min_dists)
    return d_mu, d_sigma


def pol2cart(theta, r):
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return(x, y)


def get_xy_from_vector(theta, r):
    """Convert home vector to cartesian coordinates."""
    x, y = pol2cart(theta, r)
    return x, y


def compute_location_estimate(cpu4_snapshot, cx):
    theta, r = cx.decode_cpu4(cpu4_snapshot)
    x_pred, y_pred = get_xy_from_vector(theta, r)
    return x_pred, y_pred


def compute_location_estimates(cpu4_snapshot, cx):
    x_pred = []
    y_pred = []
    for cpu4 in cpu4_snapshot:
        a, b = compute_location_estimate(cpu4, cx)
        x_pred.append(a)
        y_pred.append(b)
    return np.array(x_pred), np.array(y_pred)


def compute_estimate_error(V, T_outbound, cpu4_snapshot, cx):
    """Get distance between estimate and turning point."""
    X, Y = get_xy_from_velocity(V)
    x_actual = X[:, T_outbound]
    y_actual = Y[:, T_outbound]
    x_pred, y_pred = compute_location_estimates(cpu4_snapshot, cx)
    distances = np.sqrt((x_actual - np.array(x_pred))**2 +
                        (y_actual - np.array(y_pred))**2)
    d_mu = np.mean(distances)
    d_sigma = np.std(distances)
    return d_mu, d_sigma


def compute_relative_estimate_error(V, T_outbound, cpu4_snapshot, cx):
    """Get the distance of the estimate at turning point relative to
    total distance from nest."""
    X, Y = get_xy_from_velocity(V)
    x_actual = X[:, T_outbound]
    y_actual = Y[:, T_outbound]
    actual_dist = np.sqrt(x_actual**2 + y_actual**2)

    x_pred, y_pred = compute_location_estimates(cpu4_snapshot, cx)
    pred_dist = np.sqrt((x_actual - np.array(x_pred))**2 +
                        (y_actual - np.array(y_pred))**2)

    d_mu = np.mean(pred_dist / actual_dist)
    d_sigma = np.std(pred_dist / actual_dist)
    return d_mu, d_sigma


def compute_disk_leaving_angle(V, T_outbound, radius=20):
    X, Y = get_xy_from_velocity(V)
    N = X.shape[0]
    x_turning_point = X[:, T_outbound]
    y_turning_point = Y[:, T_outbound]
    dist_from_tp = np.sqrt((X[:, T_outbound:].T - x_turning_point)**2 +
                           (Y[:, T_outbound:].T - y_turning_point)**2)
    # Find the first point where we are distance of radius from turning point
    leaving_point = np.argmax(dist_from_tp > radius, axis=0) + T_outbound
    nest_angles = np.arctan2(-X[:, T_outbound], -Y[:, T_outbound])
    return_angles = np.arctan2(X[range(N), leaving_point] - X[:, T_outbound],
                               Y[range(N), leaving_point] - Y[:, T_outbound])
    return angular_distance(nest_angles, return_angles)


def compute_estimate_angle(V, T_outbound, cpu4_snapshot, cx):
    X, Y = get_xy_from_velocity(V)
    x_actual = X[:, T_outbound]
    y_actual = Y[:, T_outbound]
    x_pred, y_pred = compute_location_estimates(cpu4_snapshot, cx)
    theta_actual = np.arctan2(x_actual, y_actual)
    theta_pred = np.arctan2(x_pred, y_pred)
    return angular_distance(theta_actual, theta_pred)
