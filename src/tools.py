import cv2
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def Transform_Point(point: tuple[int, int, int, int], from_shape: tuple[int, int], to_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = point

    from_width, from_height = from_shape
    to_width, to_height = to_shape

    dw = to_width / from_width
    dh = to_height / from_height

    return (
        int(x1 * dw),
        int(y1 * dh),
        int(x2 * dw),
        int(y2 * dh),
    )


def Cut_New_Coor(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    dh = 40
    _y1, _y2 = 0, 0

    if 0 < y1 - dh and y2 + dh < h:
        _y1, _y2 = y1 - dh, y2 + dh
    elif y1 - dh < 0:
        _y1, _y2 = 0, y2 - y1 + 2 * dh
    else:
        _y1, _y2 = y1 - 2 * dh, h

    _x1, _x2 = 0, 0
    dw = int(1.5 * dh + 0.57 * (_y2 - _y1) - 0.5 * (x2 - x1))
    if 0 < x1 - dw and x2 + dw < w:
        _x1, _x2 = x1 - dw, x2 + dw
    elif x1 - dw < 0:
        _x1, _x2 = 0, x2 - x1 + 2 * dw
    else:
        _x1, _x2 = x2 - 2 * dw, w

    return _x1, _y1, _x2, _y2


def Interpolate(data: list[dict]) -> list[dict]:
    df = pd.DataFrame(data=data)
    df.interpolate(method='linear', inplace=True)
    df_int = df.astype(int)

    window_size = 2  # Window size must be a positive odd number
    poly_order = 1  # Polynomial order to fit in each window

    for column in ['x1', 'y1', 'x2', 'y2']:
        df[column] = savgol_filter(
            df[column], window_length=window_size, polyorder=poly_order)

    # Convert DataFrame to list of dictionaries
    interpolated_data = df_int.to_dict(orient='records')
    return interpolated_data


def Create_Interpolate_Person_Coors(frame_ids: list[int], people: list[dict]):
    data: list[dict] = []
    index: int = 0
    for frame_id in frame_ids:
        x1, y1, x2, y2 = None, None, None, None

        if index < len(people):
            person = people[index]
            if person['frame_id'] == frame_id and person['data']:
                index += 1
                x1, y1, x2, y2 = person['data'][0]
        data.append({'frame_id': frame_id, 'x1': x1,
                     'y1': y1, 'x2': x2, 'y2': y2, })

    return Interpolate(data=data)


def Interpolate_Person_Coordinates_V1(coors: list[dict]) -> list[dict]:
    data: list[dict] = []

    frame_from = coors[0]['frame_id']
    frame_to = coors[-1]['frame_id']
    index = 0
    for frame_id in range(frame_from, frame_to + 1, 1):
        if frame_id == coors[index]['frame_id']:
            data.append(coors[index])
            index += 1
        else:
            data.append({'frame_id': frame_id,
                         'x1': None, 'y1': None,
                         'x2': None, 'y2': None})

    return Interpolate(data=data)


def Interpolate_Person_Coordinates_V2(people: dict[int, dict[int, dict]], frame_ids: list[int], suspicious: set[int]) -> dict[int, dict]:
    first_frame_id: int = list(people.keys())[0]
    while frame_ids[0] < first_frame_id:
        frame_ids.pop(0)

    info: dict[int, list[dict]] = {person_id: [] for person_id in suspicious}
    for frame_id in frame_ids:
        if frame_id in people:
            for person_id in people[frame_id].keys():
                if person_id in suspicious:
                    df = {'frame_id': frame_id, **people[frame_id][person_id]}
                    info[person_id].append(df)
        else:
            for person_id in suspicious:
                if info[person_id]:
                    info[person_id].append(
                        {'frame_id': frame_id, 'x1': None, 'y1': None, 'x2': None, 'y2': None})

    all_data = {}
    for person_id in info.keys():
        data = Interpolate(data=info[person_id])
        for dt in data:
            frame_id = dt['frame_id']
            coor = {'x1': dt['x1'], 'y1': dt['y1'],
                    'x2': dt['x2'], 'y2': dt['y2']}
            all_data.setdefault(frame_id, {}).update({person_id: coor})
    return all_data


def interpolate_person_coordinates(coors: list[dict]) -> list[dict]:
    data = []

    frame_from = coors[0]['frame_id']
    frame_to = coors[-1]['frame_id']
    index = 0
    for frame_id in range(frame_from, frame_to + 1, 1):
        if index < len(coors) and frame_id == coors[index]['frame_id']:
            data.append(coors[index])
            index += 1
        else:
            data.append({'frame_id': frame_id, 'x1': None,
                        'y1': None, 'x2': None, 'y2': None})

    df = pd.DataFrame(data=data)
    df_int = df.copy()

    # Kalman Filter Initialization
    # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
    kalman = cv2.KalmanFilter(4, 2)

    # Transition matrix (A)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    # Measurement matrix (H)
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)

    # Process noise covariance (Q)
    kalman.processNoiseCov = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

    # Measurement noise covariance (R)
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.1

    for i, row in df_int.iterrows():
        if not pd.isnull(row[['x1', 'y1', 'x2', 'y2']]).all():
            # If the coordinates are available, update the Kalman filter with the measurement
            measurement = np.array(
                [row['x1'], row['y1'], row['x2'], row['y2']], dtype=np.float32).reshape((1, 4))
            kalman.correct(measurement)

        # Predict the next state
        prediction = kalman.predict()

        # Update the DataFrame with the predicted coordinates
        df_int.loc[i, ['x1', 'y1', 'x2', 'y2']
                   ] = prediction.ravel().astype(int)

    # Apply Savitzky-Golay filter for smoothing
    window_size = 3
    poly_order = 2
    for column in ['x1', 'y1', 'x2', 'y2']:
        df_int[column] = savgol_filter(
            df_int[column], window_length=window_size, polyorder=poly_order)

    interpolated_data = df_int.to_dict(orient='records')
    return interpolated_data
