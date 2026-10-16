package terrain

import "math"

// Go twin of src/shared/geo.rs (and src/skylens_model/utils/geo.py): the same
// equirectangular small-area approximation and the same Earth radius. Change
// one, change the others.

// EarthRadiusM must match EARTH_RADIUS_M in geo.rs.
const EarthRadiusM = 6378137.0

const deg = math.Pi / 180

// geoPoint is a GPS position: degrees and metres.
type geoPoint struct {
	Lat, Lon, Alt float64
}

// enu is a local East/North/Up offset in metres from an anchor.
type enu struct {
	E, N, U float64
}

// gpsToENU is gps_to_enu from geo.rs.
func gpsToENU(g, anchor geoPoint) enu {
	dLat := (g.Lat - anchor.Lat) * deg
	dLon := (g.Lon - anchor.Lon) * deg
	return enu{
		E: dLon * EarthRadiusM * math.Cos(anchor.Lat*deg),
		N: dLat * EarthRadiusM,
		U: g.Alt - anchor.Alt,
	}
}

// enuToGPS is enu_to_gps from geo.rs.
func enuToGPS(v enu, anchor geoPoint) geoPoint {
	return geoPoint{
		Lat: anchor.Lat + (v.N/EarthRadiusM)/deg,
		Lon: anchor.Lon + (v.E/(EarthRadiusM*math.Cos(anchor.Lat*deg)))/deg,
		Alt: anchor.Alt + v.U,
	}
}
