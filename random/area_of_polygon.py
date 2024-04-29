# %%
import numpy as np

# %%
A = (3.137177, 101.621620)
B = (3.137308, 101.623242)
C = (3.133062, 101.623909)
D = (3.133239, 101.621525)

lat = [A[0], B[0], C[0], D[0]]
lng = [A[1], B[1], C[1], D[1]]


# %%
def sinusoidal_projection(lat, lng):
    R = 6371009
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)

    lat_m = [R * y for y in lat_rad]
    lng_m = [R * x * np.cos(y) for y, x in zip(lat_rad, lng_rad)]

    return lat_m, lng_m


# %%
def shoelace_formula(x, y):
    return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# %%
lat_m, lng_m = sinusoidal_projection(lat, lng)
shoelace_formula(lng_m, lat_m)


# %%
def unknown_projection(lat, lng):
    R = 6371009
    lat_rad = np.radians(lat)
    lng_rad = np.radians(lng)

    lat_m = [R * np.log(np.tan(np.pi / 4 + y / 2)) for y in lat_rad]
    lng_m = [R * x for x in lng_rad]

    return lat_m, lng_m


# %%
lat_m, lng_m = unknown_projection(lat, lng)
shoelace_formula(lng_m, lat_m)

# %%
