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


def Create_Inretpolate_Person_Coors(frame_ids: list[int], people: list[dict]):
    print(frame_ids)
    print(people)

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

    df = pd.DataFrame(data=data)
    df.interpolate(method='linear', inplace=True)

    window_size = 5  # Window size must be a positive odd number
    poly_order = 2  # Polynomial order to fit in each window

    for column in ['x1', 'y1', 'x2', 'y2']:
        df[column] = savgol_filter(
            df[column], window_length=window_size, polyorder=poly_order)

    interpolated_data = []
    for _, row in df.iterrows():
        frame_id = row['frame_id']
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        if not any(np.isnan([x1, y1, x2, y2])):
            # Convert x1, y1, x2, y2 to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            interpolated_data.append(
                {'frame_id': frame_id, 'data': [(x1, y1, x2, y2)]})

    return interpolated_data
