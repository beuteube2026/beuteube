```python
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader

# =============================
# 1) Carte
# =============================
fig, ax = plt.subplots(
    figsize=(10, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

ax.set_extent([10, 30, 4, 25], crs=ccrs.PlateCarree())

# =============================
# 2) Fond général
# =============================
ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black')
ax.add_feature(cfeature.BORDERS, linewidth=1.2)
ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='gray')

# =============================
# 3) ✅ FOND BLEU CIEL DU TCHAD
# =============================
shp_path = shapereader.natural_earth(
    resolution='50m',
    category='cultural',
    name='admin_0_countries'
)

reader = shapereader.Reader(shp_path)

for record in reader.records():
    if record.attributes['NAME_LONG'] == 'Chad':
        ax.add_geometries(
            [record.geometry],
            crs=ccrs.PlateCarree(),
            facecolor='lightskyblue',
            edgecolor='black',
            linewidth=1.8,
            zorder=2
        )

# =============================
# 4) ✅ LIMITES ADMINISTRATIVES INTERNES (régions)
# =============================
shp_admin1 = shapereader.natural_earth(
    resolution='50m',
    category='cultural',
    name='admin_1_states_provinces'
)

reader_admin1 = shapereader.Reader(shp_admin1)

for region in reader_admin1.records():
    if region.attributes.get('admin') == 'Chad':
        ax.add_geometries(
            [region.geometry],
            crs=ccrs.PlateCarree(),
            facecolor='none',
            edgecolor='darkblue',
            linewidth=0.8,
            zorder=3
        )
# =============================
# 5) Ville d’étude
# =============================
ax.plot(
    15.0444, 12.1067,
    marker='o',
    color='red',
    markersize=12,
    transform=ccrs.PlateCarree()
)

ax.text(
    15.5, 11.5,
    "N'Djamena",
    fontsize=10,
    transform=ccrs.PlateCarree()
)

# =============================
# 6) ✅ NOMS DES PAYS (AVANT plt.show)
# =============================
countries = {
    "Libya": (19, 22.5),
    "Niger": (11.5, 18.5),
    "Nigeria": (12.2, 11.8),
    "Cameroon": (14, 8.5),
    "Central African Republic": (20.5, 7.3),
    "Sudan": (26.5, 14.5)
}

for country, (lon, lat) in countries.items():
    ax.text(
        lon,
        lat,
        country,
        fontsize=11 if country == "CHAD" else 10,
        fontweight='bold' if country == "CHAD" else 'normal',
        ha='center',
        va='center',
        color='black',
        transform=ccrs.PlateCarree()
    )

plt.show()

```


```python
# ================================================================
# CARTES SAISONNIÈRES DE L'ANGSTROM EXPONENT (AOD)
# MODIS Deep Blue
# Domaine : 12°E–24°E ; 6°N–24°N
# ================================================================

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# ================================================================
# 1. FICHIERS NETCDF
# ================================================================

file_paths = {
    "(a) Winter (DJF)": r"C:\Users\NOUNOU\Downloads\g4.timeAvg.MOD08_M3_6_1_Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean_Mean.20040101-20251231.SEASON_DJF.12E_6N_24E_24N.nc",

    "(b) Spring (MAM)": r"C:\Users\NOUNOU\Downloads\g4.timeAvg.MOD08_M3_6_1_Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean_Mean.20040101-20251231.SEASON_MAM.12E_6N_24E_24N.nc",

    "(c) Summer (JJA)": r"C:\Users\NOUNOU\Downloads\g4.timeAvg.MOD08_M3_6_1_Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean_Mean.20040101-20251231.SEASON_JJA.12E_6N_24E_24N.nc",

    "(d) Autumn (SON)": r"C:\Users\NOUNOU\Downloads\g4.timeAvg.MOD08_M3_6_1_Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean_Mean.20040101-20251231.SEASON_SON.12E_6N_24E_24N.nc"
}

# ================================================================
# 2. PARAMÈTRES GRAPHIQUES
# ================================================================

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ================================================================
# 3. CHARGEMENT DES DONNÉES
# ================================================================

seasonal_means = {}

for season, file_path in file_paths.items():

    print(f"\nChargement : {season}")
    print(file_path)

    with xr.open_dataset(file_path) as ds:

        print("Variables disponibles :", list(ds.data_vars))

        # --------------------------------------------------------
        # Identifier automatiquement la variable principale
        # --------------------------------------------------------
        data_vars = list(ds.data_vars.keys())

        if len(data_vars) == 0:
            raise ValueError(f"Aucune variable trouvée dans {file_path}")

        var_name = data_vars[0]

        aod_data = ds[var_name]

        # --------------------------------------------------------
        # Moyenne temporelle si la dimension time existe
        # --------------------------------------------------------
        if "time" in aod_data.dims:
            aod_data = aod_data.mean(dim="time", skipna=True)

        seasonal_means[season] = aod_data.load()

        print(f"Variable utilisée : {var_name}")
        print(f"Dimensions : {aod_data.dims}")


# ================================================================
# 4. NORMALISATION DES COORDONNÉES
# ================================================================

for season in seasonal_means:

    data = seasonal_means[season]

    # ------------------------------------------------------------
    # Vérification des coordonnées
    # ------------------------------------------------------------

    if "lon" not in data.coords:
        raise ValueError(
            f"La coordonnée 'lon' est absente pour {season}"
        )

    if "lat" not in data.coords:
        raise ValueError(
            f"La coordonnée 'lat' est absente pour {season}"
        )

    # ------------------------------------------------------------
    # Conversion des longitudes 0–360 vers -180–180
    # ------------------------------------------------------------

    if np.any(data.lon.values > 180):

        data = data.assign_coords(
            lon=((data.lon + 180) % 360) - 180
        )

        data = data.sortby("lon")

    # ------------------------------------------------------------
    # Trier les latitudes
    # ------------------------------------------------------------

    data = data.sortby("lat")

    seasonal_means[season] = data


# ================================================================
# 5. CALCUL DE L'ÉCHELLE COMMUNE
# ================================================================

all_values = []

for data in seasonal_means.values():

    values = data.values

    values = values[np.isfinite(values)]

    if values.size > 0:
        all_values.extend(values.ravel())


all_values = np.asarray(all_values)

# ---------------------------------------------------------------
# Limites de couleur
# ---------------------------------------------------------------

vmin = np.nanmin(all_values)

# Maximum fixé à 1
vmax = 1.0

print("\n================================================")
print("Échelle commune")
print("================================================")
print(f"AOD minimum = {vmin:.3f}")
print(f"AOD maximum = {vmax:.3f}")


# ================================================================
# 6. CRÉATION DE LA FIGURE
# ================================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 20),
    subplot_kw={
        "projection": ccrs.PlateCarree()
    },
    gridspec_kw={
        "wspace": 0.02,
        "hspace": 0.08
    }
)

axes = axes.ravel()

#fig, axes = plt.subplots(2, 2, figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={'wspace':0.,'hspace':0.10,
                                    #'top':1., 'bottom':0., 'left':0., 'right':1.},)
#axes = axes.ravel()
# ================================================================
# 7. ORDRE DES SAISONS
# ================================================================

season_order = [
    "(a) Winter (DJF)",
    "(b) Spring (MAM)",
    "(c) Summer (JJA)",
    "(d) Autumn (SON)"
]


# ================================================================
# 8. TRACÉ DES QUATRE CARTES
# ================================================================

im = None

for i, season in enumerate(season_order):

    ax = axes[i]

    data = seasonal_means[season]

    # ------------------------------------------------------------
    # Interpolation pour obtenir une représentation plus lisse
    # ------------------------------------------------------------

    lon_new = np.linspace(
        float(data.lon.min()),
        float(data.lon.max()),
        250
    )

    lat_new = np.linspace(
        float(data.lat.min()),
        float(data.lat.max()),
        250
    )

    data_interp = data.interp(
        lon=lon_new,
        lat=lat_new,
        method="linear"
    )

    # ------------------------------------------------------------
    # Masquage des valeurs invalides
    # ------------------------------------------------------------

    values = np.ma.masked_invalid(data_interp.values)

    lon = data_interp.lon.values
    lat = data_interp.lat.values

    lon2d, lat2d = np.meshgrid(lon, lat)

    # ============================================================
    # FOND TOPOGRAPHIQUE
    # ============================================================

    # Image satellite/topographique de Cartopy
    # On la place AVANT le champ scientifique
    try:
        ax.stock_img()
    except Exception:
        pass

    # ============================================================
    # CHAMP AOD
    # ============================================================

    im = ax.pcolormesh(
        lon2d,
        lat2d,
        values,
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=3
    )

    # ============================================================
    # LIMITES DE LA CARTE
    # ============================================================

    ax.set_extent(
        [13.5, 24, 7.5, 24],
        crs=ccrs.PlateCarree()
    )

    # ============================================================
    # ÉLÉMENTS GÉOGRAPHIQUES
    # ============================================================

    # Frontières nationales
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=2.5,
        edgecolor="black",
        zorder=5
    )

    # Côtes
    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=2.5,
        edgecolor="gray",
        zorder=5
    )

    # Cours d'eau
    ax.add_feature(
        cfeature.RIVERS,
        linewidth=2.5,
        edgecolor="blue",
        zorder=5
    )

    # ============================================================
    # GRILLE
    # ============================================================

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1.5,
        color="gray",
        alpha=1,
        linestyle="--"
    )

    # Labels
    gl.top_labels = False
    gl.right_labels = False

    # Longitude uniquement sur les cartes du bas
    if i in [0, 1]:
        gl.bottom_labels = False

    # Latitude uniquement à gauche
    if i in [1, 3]:
        gl.left_labels = False

    gl.xlabel_style = {
        "size": 12
    }

    gl.ylabel_style = {
        "size": 12
    }

    # ============================================================
    # TITRE
    # ============================================================

    ax.set_title(
        season,
        fontsize=14,
        fontweight="bold",
        pad=8
    )


# ================================================================
# 9. BARRE DE COULEUR COMMUNE
# ================================================================

# Position de la barre verticale
cax = fig.add_axes(
    [0.92, 0.15, 0.025, 0.70]
)

cb = fig.colorbar(
    im,
    cax=cax,
    orientation="vertical",
    extend="max"
)

cb.set_label(
    "(AOD550nm)_2004-2025",
    fontsize=13,
    fontweight="bold",
    labelpad=12
)

cb.ax.tick_params(
    labelsize=12,
    length=6
)


# ================================================================
# 10. TITRE GÉNÉRAL
# ================================================================

#fig.suptitle(
   # "Seasonal Distribution of Aerosol Angström Exponent\n"
    #"MODIS Deep Blue",
    #fontsize=16,
    #fontweight="bold",
    #y=0.98
#)


# ================================================================
# 11. AJUSTEMENT DE LA FIGURE
# ================================================================

plt.subplots_adjust(
    left=0.04,
    right=0.90,
    bottom=0.06,
    top=0.92,
    wspace=0.03,
    hspace=0.05
)


# ================================================================
# 12. SAUVEGARDE
# ================================================================

output_png = (
    r"C:\Users\NOUNOU\Downloads\BEUTEUBE_Aerosols.png"
)

output_pdf = (
    r"C:\Users\NOUNOU\Downloads\BEUTEUBE_Aerosols.pdf"
)

plt.savefig(
    output_png,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    output_pdf,
    dpi=300,
    bbox_inches="tight"
)


# ================================================================
# 13. AFFICHAGE
# ================================================================

plt.show()

print("\n================================================")
print("Figures enregistrées avec succès :")
print(output_png)
print(output_pdf)
print("================================================")
```


```python
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 1. Fichiers par ville
# =========================
files = {
    "N'Djamena": {
        354: r"C:\Users\NOUNOU\Downloads\NDJ_OMAERUVd_003_FinalAerosolSingleScattAlb354.20040101-20251231.12E_15N_12E_15N.nc",
        388: r"C:\Users\NOUNOU\Downloads\NDJ_OMAERUVd_003_FinalAerosolSingleScattAlb388.20040101-20251231.12E_15N_12E_15N.nc",
        500: r"C:\Users\NOUNOU\Downloads\NDJ_OMAERUVd_003_FinalAerosolSingleScattAlb500.20040101-20251231.12E_15N_12E_15N.nc"
    }
}

# =========================
# 2. Fonction lecture SSA
# =========================
def read_ssa_mean(file_path, wavelength):

    try:
        nc = Dataset(file_path)

        var_name = f"OMAERUVd_003_FinalAerosolSingleScattAlb{wavelength}"

        if var_name not in nc.variables:
            return np.nan

        var = nc.variables[var_name]
        data = var[:].astype(float)

        fill_value = getattr(var, "_FillValue", None)
        if fill_value is not None:
            data[data == fill_value] = np.nan

        scale_factor = getattr(var, "scale_factor", 1.0)
        add_offset = getattr(var, "add_offset", 0.0)
        data = data * scale_factor + add_offset

        data[(data < 0) | (data > 1)] = np.nan

        mean_ssa = np.nanmean(data)

        nc.close()
        return mean_ssa

    except:
        return np.nan


# =========================
# 3. Calcul spectres
# =========================
spectra = {}
intervals = {}

for city, wavelengths in files.items():

    wl_values = []
    ssa_values = []

    for wl in sorted(wavelengths.keys()):

        file_path = wavelengths[wl]

        if not os.path.exists(file_path):
            continue

        mean_ssa = read_ssa_mean(file_path, wl)

        wl_values.append(wl)
        ssa_values.append(mean_ssa)

    spectra[city] = (wl_values, ssa_values)

    if len(ssa_values) > 0:
        intervals[city] = (np.nanmin(ssa_values), np.nanmax(ssa_values))


# =========================
# 4. Tracé graphique
# =========================
plt.figure(figsize=(7,4), dpi=300)

for city, (wl, ssa) in spectra.items():
    plt.plot(wl, ssa, marker='o', linewidth=2.5, label=city)

plt.xlabel("wavelength_(nm)", fontsize=12,fontweight="bold" )
plt.ylabel("SSA", fontsize=12,fontweight="bold")

plt.grid(True, linestyle="--", alpha=0.8)
plt.legend()
plt.tight_layout()

plt.savefig("Spectre_SSA_publication.png", dpi=300, bbox_inches='tight')

plt.show()
```


```python

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. Charger les fichiers
# ==================================================

file_aod = r"C:\Users\NOUNOU\Downloads\AOT_NDjam_2004-2025_MERRA-2.nc"
file_ang = r"C:\Users\NOUNOU\Downloads\AET_NDjam_2004-2025_MERRA-2.nc"

ds_aod = xr.open_dataset(file_aod)
ds_ang = xr.open_dataset(file_ang)

aod = ds_aod["M2TMNXAER_5_12_4_TOTEXTTAU"]
ang = ds_ang["M2TMNXAER_5_12_4_TOTANGSTR"]

# ==================================================
# 2. Moyennes mensuelles
# ==================================================

aod_month = aod.groupby("time.month").mean()
ang_month = ang.groupby("time.month").mean()

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ==================================================
# 3. Figure
# ==================================================

fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)

# ==================================================
# 4. Courbe AOD
# ==================================================

ax1.plot(
    aod_month["month"],
    aod_month,
    marker="o",
    linewidth=2,
    color="darkred",
    label="AOT",
    zorder=3
)

ax1.set_xlabel(
    "Month",
    fontsize=12,
    fontweight="bold"
)

ax1.set_ylabel(
    "Total Aerosol Extinction_550 nm",
    color="darkred",
    fontsize=12,
    fontweight="bold"
)

ax1.tick_params(
    axis="y",
    labelcolor="darkred",
    labelsize=12
)

# ==================================================
# 5. Axe X
# ==================================================

ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(
    months,
    fontsize=12,
    fontweight="bold"
)

# ==================================================
# 6. Courbe Angström
# ==================================================

ax2 = ax1.twinx()

ax2.plot(
    ang_month["month"],
    ang_month,
    marker="s",
    linewidth=2,
    color="blue",
    label="Angström",
    zorder=4
)

ax2.set_ylabel(
    "Total Aerosol Angström_(470–870 nm)",
    color="blue",
    fontsize=12,
    fontweight="bold"
)

ax2.tick_params(
    axis="y",
    labelcolor="blue",
    labelsize=12
)

# ==================================================
# 7. CORRECTION DE LA GRILLE VERTICALE
# ==================================================

# Mettre l'axe secondaire derrière l'axe principal
ax2.set_zorder(0)
ax1.set_zorder(1)

# Rendre le fond de ax1 transparent
ax1.patch.set_visible(False)

# Grille verticale sur les 12 mois
ax1.grid(
    axis="x",
    which="major",
    linestyle="--",
    linewidth=0.8,
    alpha=1,
    zorder=0
)

# ==================================================
# 8. Limites de l'axe X
# ==================================================

ax1.set_xlim(0.5, 12.5)

# ==================================================
# 9. Mise en page
# ==================================================

plt.tight_layout()

plt.show()

```


```python
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# ==================================================
# 1. Charger les fichiers
# ==================================================

file_aod = r"C:\Users\NOUNOU\Downloads\AOT_NDjam_2004-2025_MERRA-2.nc"
file_alb = r"C:\Users\NOUNOU\Downloads\Njam.OMAERUVd_003_FinalAerosolSingleScattAlb500.20000101-20251231.12E_15N_12E_15N.nc"
file_ang = r"C:\Users\NOUNOU\Downloads\AET_NDjam_2004-2025_MERRA-2.nc"

ds_aod = xr.open_dataset(file_aod)
ds_alb = xr.open_dataset(file_alb)
ds_ang = xr.open_dataset(file_ang)

# Variables
aod = ds_aod["M2TMNXAER_5_12_4_TOTEXTTAU"]
alb = ds_alb["OMAERUVd_003_FinalAerosolSingleScattAlb500"]
ang = ds_ang["M2TMNXAER_5_12_4_TOTANGSTR"]

# ==================================================
# 2. Moyennes mensuelles
# ==================================================

aod_month = aod.groupby("time.month").mean()
alb_month = alb.groupby("time.month").mean()
ang_month = ang.groupby("time.month").mean()

months == ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

# ==================================================
# 3. Création des deux figures côte à côte
# ==================================================

fig, (ax1, ax3) = plt.subplots(1,2, figsize=(15,5), dpi=300)

# ==================================================
# FIGURE (a) : AOD vs SSA
# ==================================================

ax1.plot(aod_month["month"], aod_month,
         marker='o', linewidth=2.5, color='darkred')

ax1.set_xlabel("Month", fontsize=12, fontweight='bold')
ax1.set_ylabel("Total Aerosol Extinction_550 nm", color='darkred', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='darkred',labelsize=13)

ax1.set_xticks(range(1,13))
ax1.set_xticklabels(months, fontsize=12, fontweight='bold')
ax2 = ax1.twinx()

ax2.plot(alb_month["month"], alb_month,
         marker='s', linewidth=2.5, color='blue')

ax2.set_ylabel("Single Scattering Albedo_500 nm", color='blue', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='blue',labelsize=13)

ax1.set_title("(a) AOT and SSA", fontsize=12, fontweight='bold')
#ax1.grid(True)
ax1.grid(axis='x', linestyle='--', alpha=1)
# ==================================================
# FIGURE (b) : Angström vs SSA
# ==================================================
# ----- Axe gauche : Angstrom -----
ax3.plot(ang_month["month"], ang_month,
         marker='s',
         linewidth=2.5,
         color='darkred')

ax3.set_xlabel("Month", fontsize=12, fontweight='bold')
ax3.set_ylabel("Total Aerosol Angstrom_(470–870 nm)", color='darkred', fontsize=12, fontweight='bold')
ax3.tick_params(axis='y', labelcolor='darkred',labelsize=12)

ax3.set_xticks(range(1,13))
ax3.set_xticklabels(months, fontsize=12, fontweight='bold')

# ----- Axe droit : SSA -----
ax4 = ax3.twinx()

ax4.plot(alb_month["month"], alb_month,
         marker='o',
         linewidth=2.5,
         color='blue')

ax4.set_ylabel("Single Scattering Albedo_500 nm", color='blue', fontsize=12, fontweight='bold')
ax4.tick_params(axis='y', labelcolor='blue',labelsize=13)

ax3.set_title("(b) AET and SSA", fontsize=12, fontweight='bold')

# ax3.grid(True)
ax3.grid(axis='x', linestyle='--', alpha=1)
plt.tight_layout()
plt.margins(x=0)
plt.show()
```


```python
# ================================================================
# CAMS — VARIABILITÉ MENSUELLE DES MOYENNES INTERANNUELLES
# DU FORÇAGE RADIATIF DES AÉROSOLS À N'DJAMENA
#
# Période : 2008–2017
#
# (a) BOA / Surface
# (b) TOA
#
# Résultat :
# 12 moyennes interannuelles mensuelles
# Janvier, Février, ..., Décembre
# ================================================================

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ================================================================
# 1. PARAMÈTRES
# ================================================================

FILE_SRF = (
    r"/media/beuteube/BERIF_DATA/ERA5_20-08-2026/"
    r"Forcage-radiatif-absorption-aerosol-AOD_2008-2017/"
    r"Radiative_forcing-dust-2008-2017/"
    r"CAMS_dust_srf_sw_cs_2008_2017.nc"
)

FILE_TOA = (
    r"/media/beuteube/BERIF_DATA/ERA5_20-08-2026/"
    r"Forcage-radiatif-absorption-aerosol-AOD_2008-2017/"
    r"Radiative_forcing-dust-2008-2017/"
    r"CAMS_dust_toa_sw_cs_2008_2017.nc"
)


# Coordonnées de N'Djamena
LAT_NDJ = 12.1348
LON_NDJ = 15.0557


# Période
YEAR_START = 2008
YEAR_END = 2017


# Mois
MONTHS = np.arange(1, 13)

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]


# Dossier de sortie
OUTPUT_DIR = (
    "Forcage_radiatif_aerosols_Ndjamena_2008_2017"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ================================================================
# 2. IDENTIFICATION DES COORDONNÉES
# ================================================================

def find_coordinate(ds, possible_names):

    for name in possible_names:

        if name in ds.coords:
            return name

        if name in ds.variables:
            return name

    raise ValueError(
        "Coordonnée introuvable parmi : "
        + str(possible_names)
    )


# ================================================================
# 3. IDENTIFICATION DE LA VARIABLE
# ================================================================

def find_data_variable(ds):

    # Cas simple : une seule variable
    if len(ds.data_vars) == 1:

        return list(ds.data_vars)[0]

    keywords = [
        "dust",
        "sw",
        "forcing",
        "cs",
        "rf"
    ]

    candidates = []

    for var in ds.data_vars:

        name = var.lower()

        score = sum(
            keyword in name
            for keyword in keywords
        )

        candidates.append(
            (score, var)
        )

    candidates.sort(
        reverse=True
    )

    if candidates[0][0] > 0:

        return candidates[0][1]

    return list(ds.data_vars)[0]


# ================================================================
# 4. DIAGNOSTIC DE LA RÉSOLUTION TEMPORELLE
# ================================================================

def analyse_resolution_temporelle(
        file_path,
        label):

    print("\n" + "=" * 75)
    print(
        f"ANALYSE TEMPORELLE : {label}"
    )
    print("=" * 75)

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Fichier introuvable :\n{file_path}"
        )

    ds = xr.open_dataset(
        file_path,
        decode_times=True
    )

    # ------------------------------------------------------------
    # Identifier la dimension temporelle
    # ------------------------------------------------------------

    time_candidates = [
        "time",
        "valid_time",
        "datetime",
        "date"
    ]

    time_name = None

    for name in time_candidates:

        if name in ds.coords:

            time_name = name
            break

        if name in ds.variables:

            time_name = name
            break

    if time_name is None:

        raise ValueError(
            "Aucune dimension temporelle trouvée."
        )

    time = ds[time_name]

    # ------------------------------------------------------------
    # Nombre de dates
    # ------------------------------------------------------------

    print(
        f"\nDimension temporelle : {time_name}"
    )

    print(
        f"Nombre total de pas : {time.size}"
    )

    # ------------------------------------------------------------
    # Premières dates
    # ------------------------------------------------------------

    print("\n10 premiers pas de temps :")

    for t in time.values[:10]:

        print("   ", t)

    # ------------------------------------------------------------
    # Intervalles temporels
    # ------------------------------------------------------------

    time_values = time.values

    if len(time_values) > 1:

        try:

            time_pd = pd.to_datetime(
                time_values
            )

            differences = (
                time_pd[1:]
                - time_pd[:-1]
            )

            seconds = np.array(
                [
                    d.total_seconds()
                    for d in differences
                ]
            )

            hours = seconds / 3600.0

            # ----------------------------------------------------
            # Statistiques
            # ----------------------------------------------------

            print("\nIntervalle temporel :")

            print(
                f"   Minimum : "
                f"{np.nanmin(hours):.3f} h"
            )

            print(
                f"   Maximum : "
                f"{np.nanmax(hours):.3f} h"
            )

            print(
                f"   Moyenne : "
                f"{np.nanmean(hours):.3f} h"
            )

            print(
                f"   Médiane : "
                f"{np.nanmedian(hours):.3f} h"
            )

            # ----------------------------------------------------
            # Intervalles uniques
            # ----------------------------------------------------

            unique_hours = np.unique(
                np.round(hours, 6)
            )

            print(
                "\nIntervalles temporels uniques :"
            )

            for h in unique_hours:

                print(
                    f"   {h:.3f} heure(s)"
                )

            # ----------------------------------------------------
            # Résolution dominante
            # ----------------------------------------------------

            values, counts = np.unique(
                np.round(hours, 6),
                return_counts=True
            )

            index = np.argmax(counts)

            dominant_resolution = (
                values[index]
            )

            print(
                "\n>>> RÉSOLUTION TEMPORELLE "
                "DOMINANTE : "
                f"{dominant_resolution:.3f} heure(s)"
            )

            print(
                f">>> Soit : "
                f"{dominant_resolution * 60:.1f} minutes"
            )

            # ----------------------------------------------------
            # Nombre théorique de données par jour
            # ----------------------------------------------------

            if dominant_resolution > 0:

                observations_per_day = (
                    24 /
                    dominant_resolution
                )

                print(
                    ">>> Nombre théorique de "
                    "données/jour : "
                    f"{observations_per_day:.2f}"
                )

            # ----------------------------------------------------
            # Anomalies temporelles
            # ----------------------------------------------------

            anomalous = (
                np.abs(
                    hours -
                    dominant_resolution
                ) > 1e-6
            )

            n_anomalous = np.sum(
                anomalous
            )

            print(
                "\nNombre d'intervalles temporels "
                f"non conformes : {n_anomalous}"
            )

            if n_anomalous > 0:

                print(
                    "\nATTENTION : "
                    "la série contient des intervalles "
                    "temporels irréguliers."
                )

        except Exception as e:

            print(
                "\nImpossible de calculer "
                "automatiquement les intervalles :"
            )

            print(e)

    # ------------------------------------------------------------
    # Période
    # ------------------------------------------------------------

    print("\nPériode couverte :")

    print(
        "   Début :",
        time_values[0]
    )

    print(
        "   Fin   :",
        time_values[-1]
    )

    # ------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------

    print("\nDimensions du fichier :")

    for dim, size in ds.sizes.items():

        print(
            f"   {dim} : {size}"
        )

    # ------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------

    print("\nVariables disponibles :")

    for var in ds.data_vars:

        print(
            f"   {var}"
        )

        units = ds[var].attrs.get(
            "units",
            "non spécifiées"
        )

        print(
            f"      unités : {units}"
        )

    ds.close()


# ================================================================
# 5. EXTRACTION DE N'DJAMENA ET CALCUL DES MOYENNES
# ================================================================

def extract_ndjamena(
        file_path,
        label):

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Fichier introuvable :\n{file_path}"
        )

    ds = xr.open_dataset(
        file_path,
        decode_times=True
    )

    print("\nVariables :")
    print(list(ds.data_vars))

    print("\nDimensions :")
    print(ds.dims)

    # ------------------------------------------------------------
    # Coordonnées
    # ------------------------------------------------------------

    lat_name = find_coordinate(
        ds,
        [
            "lat",
            "latitude",
            "Latitude",
            "LAT"
        ]
    )

    lon_name = find_coordinate(
        ds,
        [
            "lon",
            "longitude",
            "Longitude",
            "LON"
        ]
    )

    time_name = find_coordinate(
        ds,
        [
            "time",
            "valid_time",
            "datetime",
            "date"
        ]
    )

    # ------------------------------------------------------------
    # Variable
    # ------------------------------------------------------------

    var_name = find_data_variable(ds)

    da = ds[var_name]

    units = da.attrs.get(
        "units",
        "W m$^{-2}$"
    )

    print(
        f"\nVariable : {var_name}"
    )

    print(
        f"Unités   : {units}"
    )

    # ------------------------------------------------------------
    # Extraction du point le plus proche
    # ------------------------------------------------------------

    point = da.sel(
        {
            lat_name: LAT_NDJ,
            lon_name: LON_NDJ
        },
        method="nearest"
    )

    lat_selected = float(
        point[lat_name].values
    )

    lon_selected = float(
        point[lon_name].values
    )

    print("\nN'Djamena :")

    print(
        f"Coordonnées demandées : "
        f"{LAT_NDJ:.4f}°N ; "
        f"{LON_NDJ:.4f}°E"
    )

    print(
        f"Point CAMS utilisé : "
        f"{lat_selected:.4f}°N ; "
        f"{lon_selected:.4f}°E"
    )

    # ------------------------------------------------------------
    # Sélection de la période
    # ------------------------------------------------------------

    point = point.sel(
        {
            time_name: slice(
                f"{YEAR_START}-01-01",
                f"{YEAR_END}-12-31"
            )
        }
    )

    # ------------------------------------------------------------
    # Moyennes mensuelles
    #
    # Première étape :
    # moyenne de chaque mois de chaque année.
    #
    # Janvier 2008
    # Janvier 2009
    # ...
    # Janvier 2017
    # ------------------------------------------------------------

    monthly = point.resample(
        {
            time_name: "MS"
        }
    ).mean(
        skipna=True
    )

    # ------------------------------------------------------------
    # Moyenne interannuelle par mois
    #
    # Janvier =
    # moyenne des 10 mois de janvier
    #
    # Février =
    # moyenne des 10 mois de février
    #
    # ...
    # ------------------------------------------------------------

    monthly_interannual = (
        monthly
        .groupby(
            f"{time_name}.month"
        )
        .mean(
            dim=time_name,
            skipna=True
        )
    )

    # ------------------------------------------------------------
    # Conversion en tableau
    # ------------------------------------------------------------

    values = np.asarray(
        monthly_interannual.values
    ).squeeze()

    df = pd.DataFrame({

        "Month": MONTHS,

        "Month_Name": MONTH_LABELS,

        "Value": values

    })

    print(
        "\nMoyennes mensuelles interannuelles :"
    )

    print(df)

    ds.close()

    return (
        df,
        units,
        lat_selected,
        lon_selected,
        var_name
    )


# ================================================================
# 6. DIAGNOSTIC DE LA RÉSOLUTION TEMPORELLE
# ================================================================

analyse_resolution_temporelle(
    FILE_SRF,
    "CAMS — Forçage radiatif à la surface"
)

analyse_resolution_temporelle(
    FILE_TOA,
    "CAMS — Forçage radiatif au TOA"
)


# ================================================================
# 7. FORÇAGE À LA SURFACE
# ================================================================

(
    df_srf,
    units_srf,
    lat_srf,
    lon_srf,
    var_srf
) = extract_ndjamena(
    FILE_SRF,
    "FORÇAGE RADIATIF À LA SURFACE (BOA)"
)


# ================================================================
# 8. FORÇAGE AU TOA
# ================================================================

(
    df_toa,
    units_toa,
    lat_toa,
    lon_toa,
    var_toa
) = extract_ndjamena(
    FILE_TOA,
    "FORÇAGE RADIATIF AU TOA"
)


# ================================================================
# 9. TABLEAU FINAL
# ================================================================

df_final = pd.DataFrame({

    "Month": MONTHS,

    "Month_Name": MONTH_LABELS,

    "SRF": df_srf["Value"].values,

    "TOA": df_toa["Value"].values

})


print("\n" + "=" * 70)
print(
    "MOYENNES MENSUELLES INTERANNUELLES 2008–2017"
)
print("=" * 70)

print(df_final)


# ================================================================
# 10. SAUVEGARDE DU TABLEAU
# ================================================================

csv_file = os.path.join(
    OUTPUT_DIR,
    "CAMS_Ndjamena_moyennes_mensuelles_"
    "interannuelles_2008_2017.csv"
)

df_final.to_csv(
    csv_file,
    index=False
)

print(
    "\nTableau sauvegardé :"
)

print(csv_file)


# ================================================================
# 11. FIGURE — BOA ET TOA CÔTE À CÔTE
# ================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(17, 7),
    dpi=600
)


# ================================================================
# 12. PANNEAU (a) — BOA / SURFACE
# ================================================================

ax = axes[0]

ax.bar(
    MONTHS,
    df_final["SRF"],
    width=0.70,
    edgecolor="black",
    linewidth=0.8
)

# Ligne zéro
ax.axhline(
    0,
    linewidth=0.9,
    linestyle="--"
)

ax.set_xticks(
    MONTHS
)

ax.set_xticklabels(
    MONTH_LABELS,
    fontsize=14,fontweight="bold"
)

ax.set_xlabel(
    "Month",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    f"Radiative forcing ({units_srf})",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "BOA",
    fontsize=15,
    fontweight="bold",
    pad=12
)

ax.grid(
    axis="x",
    linestyle=":",
    alpha=1
)


# ================================================================
# 13. PANNEAU (b) — TOA
# ================================================================

ax = axes[1]

ax.bar(
    MONTHS,
    df_final["TOA"],
    width=0.70,
    edgecolor="black",
    linewidth=0.8
)

# Ligne zéro
ax.axhline(
    0,
    linewidth=0.9,
    linestyle="--"
)

ax.set_xticks(
    MONTHS
)

ax.set_xticklabels(
    MONTH_LABELS,
    fontsize=14,fontweight="bold"
)

ax.set_xlabel(
    "Month",
    fontsize=14,
    fontweight="bold"
)

ax.set_ylabel(
    f"Radiative forcing ({units_toa})",
    fontsize=14,
    fontweight="bold"
)

ax.set_title(
    "TOA",
    fontsize=15,
    fontweight="bold",
    pad=12
)

ax.grid(
    axis="x",
    linestyle=":",
    alpha=1
)


# ================================================================
# 14. MISE EN PAGE
# ================================================================

plt.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.90,
    bottom=0.12,
    wspace=0.20
)


# ================================================================
# 15. SAUVEGARDE DE LA FIGURE
# ================================================================

figure_file = os.path.join(
    OUTPUT_DIR,
    "CAMS_Ndjamena_variabilite_mensuelle_"
    "moyennes_interannuelles_SRF_TOA_2008_2017.png"
)

plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)


# ================================================================
# 16. AFFICHAGE
# ================================================================

for ax in [axes[0], axes[1]]:
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
ax = axes[0].margins(x=0)
ax = axes[1].margins(x=0)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("ANALYSE TERMINÉE")
print("=" * 70)

print("\nFigure :")
print(figure_file)

print("\nTableau :")
print(csv_file)
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =====================================================
# 1. LECTURE FICHIER
# =====================================================

file_path = r"C:\Users\NOUNOU\Desktop\Données\Granulométriendjam2010-Aout-sep-Oct.csv"

df = pd.read_csv(
    file_path,
    sep=";",
    decimal=",",
    encoding="latin1",
    engine="python"
)

df.columns = df.columns.str.strip()
df = df.dropna(how="all")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# =====================================================
# 2. EXTRACTION DONNÉES
# =====================================================

diam = df.iloc[:,0].values
Aout = df.iloc[:,1].values
Sep  = df.iloc[:,2].values
Octo = df.iloc[:,3].values

mask = diam > 0
diam = diam[mask]
Aout = Aout[mask]
Sep  = Sep[mask]
Octo = Octo[mask]

# =====================================================
# 3. LOGNORMALE
# =====================================================

def lognorm_number(D, N0, Dg, sigma):
    return (N0 / (D * np.log(sigma) * np.sqrt(2*np.pi))) * \
           np.exp(-(np.log(D/Dg))**2 / (2*(np.log(sigma))**2))

def fit_lognorm(D, N):
    p0 = [max(N), np.median(D), 1.8]
    bounds = ([0, min(D), 1.1], [1e6, max(D), 3.5])

    popt, _ = curve_fit(
        lognorm_number,
        D,
        N,
        p0=p0,
        bounds=bounds,
        maxfev=30000
    )

    D_smooth = np.logspace(np.log10(min(D)), np.log10(max(D)), 400)
    N_smooth = lognorm_number(D_smooth, *popt)

    return D_smooth, N_smooth, popt

# Ajustement
x1, n1, p1 = fit_lognorm(diam, Aout)
x2, n2, p2 = fit_lognorm(diam, Sep)
x3, n3, p3 = fit_lognorm(diam, Octo)

# =====================================================
# 4. TRANSFORMATIONS PHYSIQUES
# =====================================================

# Nombre
N1, N2, N3 = n1, n2, n3

# Volume
V1 = (np.pi/6) * x1**3 * n1
V2 = (np.pi/6) * x2**3 * n2
V3 = (np.pi/6) * x3**3 * n3

# Surface
S1 = np.pi * x1**2 * n1
S2 = np.pi * x2**2 * n2
S3 = np.pi * x3**2 * n3

# =====================================================
# 5. FIGURE AVEC 3 PANNEAUX
# =====================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

# ---------- (a) Nombre ----------
axes[0].plot(x1, N1, linewidth=2.5, label="August")
axes[0].plot(x2, N2, linewidth=2.5, label="September")
axes[0].plot(x3, N3, linewidth=2.5, label="October")

axes[0].set_xscale("log")
axes[0].set_title("(a) Number", fontweight='bold')
axes[0].set_xlabel("Radius (µm)", fontweight='bold')
axes[0].set_ylabel("dN/dlogr", fontweight='bold')
axes[0].grid(True, which="both", linestyle="--", alpha=0.8)
axes[0].legend()

# ---------- (b) Volume ----------
axes[1].plot(x1, V1, linewidth=2.5, label="August")
axes[1].plot(x2, V2, linewidth=2.5, label="September")
axes[1].plot(x3, V3, linewidth=2.5, label="October")

axes[1].set_xscale("log")
axes[1].set_title("(b) Volume", fontweight='bold')
axes[1].set_xlabel("Radius (µm)", fontweight='bold')
axes[1].set_ylabel("dV/dlogr", fontweight='bold')
axes[1].grid(True, which="both", linestyle="--", alpha=0.8)

# ---------- (c) Surface ----------
axes[2].plot(x1, S1, linewidth=2.5, label="August")
axes[2].plot(x2, S2, linewidth=2.5, label="September")
axes[2].plot(x3, S3, linewidth=2.5, label="October")

axes[2].set_xscale("log")
axes[2].set_title("(c) Surface", fontweight='bold')
axes[2].set_xlabel("Radius (µm)", fontweight='bold')
axes[2].set_ylabel("dS/dlogr", fontweight='bold')
axes[2].grid(True, which="both", linestyle="--", alpha=0.8)

# =====================================================
# AJUSTEMENTS
# =====================================================

plt.tight_layout()
plt.savefig("Distribution_complete_3panneaux.png", dpi=300)
plt.show()

# =====================================================
# 6. PARAMÈTRES
# =====================================================

print("\n=== PARAMÈTRES LOGNORMAUX ===")
print("August     : Dg = %.3f µm   σ = %.3f" % (p1[1], p1[2]))
print("September  : Dg = %.3f µm   σ = %.3f" % (p2[1], p2[2]))
print("October    : Dg = %.3f µm   σ = %.3f" % (p3[1], p3[2]))
```
