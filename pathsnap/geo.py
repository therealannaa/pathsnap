"""Coordinate handling.

Each trip covers a few kilometres, so we project lon/lat to a local
equirectangular plane centred on the trip. Over 20 km the scale error is well
under 0.5%, which is far below GPS noise, and it lets us use plain Euclidean
geometry (point-to-segment distance, bearings) in metres.
"""
import numpy as np

EARTH_R = 6371008.8


class LocalProjection:
    def __init__(self, lon0, lat0):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        self.kx = np.radians(1.0) * EARTH_R * np.cos(np.radians(self.lat0))
        self.ky = np.radians(1.0) * EARTH_R

    @classmethod
    def for_track(cls, track):
        return cls(np.median(track[:, 0]), np.median(track[:, 1]))

    def forward(self, lonlat):
        lonlat = np.asarray(lonlat, dtype=float)
        return np.column_stack([(lonlat[:, 0] - self.lon0) * self.kx,
                                (lonlat[:, 1] - self.lat0) * self.ky])

    def inverse(self, xy):
        xy = np.asarray(xy, dtype=float)
        return np.column_stack([xy[:, 0] / self.kx + self.lon0,
                                xy[:, 1] / self.ky + self.lat0])


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points to segments (broadcasting) and the projection fraction t in [0, 1]."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t


def bearing(dx, dy):
    """Bearing in radians, 0 = east, counter-clockwise."""
    return np.arctan2(dy, dx)
