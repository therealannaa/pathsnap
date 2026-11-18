"""Folium map building."""
import folium
import numpy as np


def _arc_lines(net, arcs):
    a = np.asarray(arcs, dtype=int)
    p0 = net.proj.inverse(np.column_stack([net.ax[a], net.ay[a]]))
    p1 = net.proj.inverse(np.column_stack([net.bx[a], net.by[a]]))
    return [[(y0, x0), (y1, x1)] for (x0, y0), (x1, y1) in zip(p0, p1)]


def _chain(net, arcs):
    """Polyline through a sequence of arcs (lat, lon)."""
    a = np.asarray(arcs, dtype=int)
    if len(a) == 0:
        return []
    pts = np.column_stack([np.r_[net.ax[a], net.bx[a[-1]]], np.r_[net.ay[a], net.by[a[-1]]]])
    ll = net.proj.inverse(pts)
    return [(lat, lon) for lon, lat in ll]


def nearby_arcs(net, pxy, radius=120.0):
    idx = net.tree.query_ball_point(pxy, r=radius)
    s = set()
    for row in idx:
        s.update(net.sample_arc[row].tolist())
    return sorted(s)


def trip_map(net, track_lonlat, pxy, ref_route_local=None, layers=None, title=None, max_network_arcs=15000):
    """layers: dict name -> (list of local arcs as a path, colour)."""
    center = [float(np.mean(track_lonlat[:, 1])), float(np.mean(track_lonlat[:, 0]))]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    near = nearby_arcs(net, pxy)[:max_network_arcs]
    fg = folium.FeatureGroup(name=f"Road network near track ({len(near)} arcs)", show=True)
    for line in _arc_lines(net, near):
        folium.PolyLine(line, color="#9e9e9e", weight=1.5, opacity=0.6).add_to(fg)
    fg.add_to(m)

    if ref_route_local is not None:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        folium.PolyLine(_chain(net, ref_route_local), color="#2e7d32", weight=7, opacity=0.6,
                        tooltip="Reference route").add_to(fg)
        fg.add_to(m)

    for name, (arcs, colour) in (layers or {}).items():
        fg = folium.FeatureGroup(name=name, show=True)
        for line in _arc_lines(net, arcs):
            folium.PolyLine(line, color=colour, weight=4, opacity=0.9, tooltip=name).add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup(name="Raw GPS observations", show=True)
    step = max(1, len(track_lonlat) // 2500)
    for lon, lat, t in track_lonlat[::step]:
        folium.CircleMarker((lat, lon), radius=2, color="#1565c0", fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={t:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.Marker((track_lonlat[0, 1], track_lonlat[0, 0]), tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((track_lonlat[-1, 1], track_lonlat[-1, 0]), tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    legend_items = [("#1565c0", "Raw GPS"), ("#9e9e9e", "Road network")]
    if ref_route_local is not None:
        legend_items.append(("#2e7d32", "Reference route"))
    legend_items += [(c, n) for n, (_, c) in (layers or {}).items()]
    rows = "".join(f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{n}</div>' for c, n in legend_items)
    head = f"<b>{title}</b><br>" if title else ""
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
        f'border:1px solid #999;border-radius:4px;font:12px sans-serif">{head}{rows}</div>'))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[track_lonlat[:, 1].min(), track_lonlat[:, 0].min()],
                  [track_lonlat[:, 1].max(), track_lonlat[:, 0].max()]])
    return m
