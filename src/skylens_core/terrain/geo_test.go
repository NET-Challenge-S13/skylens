package terrain

import (
	"math"
	"testing"
)

// Default anchor of the model server, as in geo.rs's tests.
var testAnchor = geoPoint{Lat: 36.3685, Lon: 127.3475, Alt: 30}

func TestAnchorMapsToOrigin(t *testing.T) {
	if got := gpsToENU(testAnchor, testAnchor); got != (enu{}) {
		t.Fatalf("anchor -> %+v, want origin", got)
	}
}

func TestGpsToENUMatchesGeoRs(t *testing.T) {
	// The value geo.rs's matches_geo_py_values checks:
	// e = 0.0015 deg * R * cos(36.3685 deg), n = 0.0010 deg * R, u = 42.5 - 30.
	// Literals computed independently in double precision.
	got := gpsToENU(geoPoint{Lat: 36.3695, Lon: 127.3490, Alt: 42.5}, testAnchor)
	if math.Abs(got.E-134.455008003) > 1e-6 {
		t.Errorf("e = %.9f, want 134.455008003", got.E)
	}
	if math.Abs(got.N-111.319490794) > 1e-6 {
		t.Errorf("n = %.9f, want 111.319490794", got.N)
	}
	if math.Abs(got.U-12.5) > 1e-12 {
		t.Errorf("u = %g, want 12.5", got.U)
	}
	// And the formula itself, written out as in geo.rs.
	e := 0.0015 * deg * EarthRadiusM * math.Cos(36.3685*deg)
	n := 0.0010 * deg * EarthRadiusM
	if math.Abs(got.E-e) > 1e-6 || math.Abs(got.N-n) > 1e-6 {
		t.Errorf("formula mismatch: %+v vs e=%g n=%g", got, e, n)
	}
}

func TestENURoundTrips(t *testing.T) {
	g := geoPoint{Lat: 36.3712, Lon: 127.3441, Alt: 55}
	back := enuToGPS(gpsToENU(g, testAnchor), testAnchor)
	if math.Abs(back.Lat-g.Lat) > 1e-12 || math.Abs(back.Lon-g.Lon) > 1e-12 || math.Abs(back.Alt-g.Alt) > 1e-12 {
		t.Fatalf("round trip %+v -> %+v", g, back)
	}
}
