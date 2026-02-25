import pyproj


def get_proj_wgs84():
    return pyproj.CRS("EPSG:4326")

def get_proj_aeqd(lat, lon):
    return pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84")

def get_transformer_wgs84_to_aeqd(ref_lat, ref_lon):
    return pyproj.Transformer.from_crs(get_proj_wgs84(), get_proj_aeqd(ref_lat, ref_lon), always_xy=True)

def get_transformer_aeqd_to_wgs84(ref_lat, ref_lon):
    return pyproj.Transformer.from_crs(get_proj_aeqd(ref_lat, ref_lon), get_proj_wgs84(), always_xy=True)