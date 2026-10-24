package terrain

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestTileMathKnownValues(t *testing.T) {
	if got := lonToTileX(0, 1); got != 1 {
		t.Errorf("lon 0 z1 -> %g, want 1", got)
	}
	if got := latToTileY(0, 1); math.Abs(got-1) > 1e-12 {
		t.Errorf("lat 0 z1 -> %g, want 1", got)
	}
	// Daejeon (127.3475 E, 36.3685 N) lies in z12 tile 3496/1603.
	if x := int(math.Floor(lonToTileX(127.3475, 12))); x != 3496 {
		t.Errorf("z12 x = %d, want 3496", x)
	}
	if y := int(math.Floor(latToTileY(36.3685, 12))); y != 1603 {
		t.Errorf("z12 y = %d, want 1603", y)
	}
	for _, lat := range []float64{-60, 0, 36.3685, 37.5} {
		if back := tileYToLat(latToTileY(lat, 14), 14); math.Abs(back-lat) > 1e-9 {
			t.Errorf("lat %g round trip -> %g", lat, back)
		}
	}
	if back := tileXToLon(lonToTileX(127.3475, 14), 14); math.Abs(back-127.3475) > 1e-9 {
		t.Errorf("lon round trip -> %g", back)
	}
}

func TestTerrariumDecoding(t *testing.T) {
	if got := terrariumElevation(128, 0, 0); got != 0 {
		t.Errorf("(128,0,0) -> %g m, want 0", got)
	}
	if got := terrariumElevation(128, 100, 128); got != 100.5 {
		t.Errorf("(128,100,128) -> %g m, want 100.5", got)
	}
	img := image.NewNRGBA(image.Rect(0, 0, 256, 256))
	img.SetNRGBA(3, 2, color.NRGBA{R: 129, G: 0, B: 0, A: 255})
	e := decodeTerrarium(img)
	if e[2*256+3] != 256 {
		t.Errorf("pixel (3,2) = %g, want 256", e[2*256+3])
	}
	if e[0] != -32768 {
		t.Errorf("pixel (0,0) = %g, want -32768", e[0])
	}
}

func TestTilesCoveringIsInclusive(t *testing.T) {
	// A box straddling the corner shared by z12 tiles 3497..3498 / 1612..1613.
	lon := tileXToLon(3498, 12)
	lat := tileYToLat(1613, 12)
	keys := tilesCovering(lat-0.001, lon-0.001, lat+0.001, lon+0.001, 12)
	if len(keys) != 4 {
		t.Fatalf("got %d tiles, want 4: %v", len(keys), keys)
	}
	if keys[0] != (tileKey{12, 3497, 1612}) || keys[3] != (tileKey{12, 3498, 1613}) {
		t.Errorf("keys %v", keys)
	}
}

func TestSanitizeGridRepairsVoids(t *testing.T) {
	nan := math.NaN()
	h := []float64{
		10, 10, 10,
		10, nan, 10,
		-32768, 10, 10,
	}
	if !sanitizeGrid(h, 3, 3) {
		t.Fatal("grid reported unrecoverable")
	}
	for i, v := range h {
		if v != 10 {
			t.Errorf("h[%d] = %g, want 10", i, v)
		}
	}
	if sanitizeGrid([]float64{nan, nan}, 2, 1) {
		t.Error("all-void grid should be unrecoverable")
	}
}
