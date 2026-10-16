package terrain

import (
	"math"
	"sort"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
)

// Cells are squares of a fixed ENU grid about the anchor: cell (x, y) covers
// east [x*CellSizeM, (x+1)*CellSizeM) and north [y*CellSizeM, (y+1)*CellSizeM).
// Because the grid hangs off the fixed anchor, the same cell has the same
// block id in every stream.
const CellSizeM = 400.0

// RingSpan: the background ring covers the area's span times this, centred on
// the area (legacy RING_SPAN).
const RingSpan = 3.0

// Caps against absurd areas (a 400 m cell grid over a province).
const (
	maxCoreCells = 4096
	maxRingCells = 2048
)

// profile is the resolution of one detail level.
type profile struct {
	level modelv1.Level
	// ring marks the coarse background ring.
	ring bool
	// gridStep is the spacing of the height grid in metres. The grid has a
	// vertex on every cell edge, so neighbouring cells share their border
	// heights and the screen can stitch them without seams.
	gridStep float64
	// satZoom is the VWorld imagery zoom the cell image is resampled from.
	satZoom int
	// imagePx is the edge of the cell image in pixels.
	imagePx int
}

var (
	// fineProfile is the area itself: a 10 m height grid (the DEM is ~30 m)
	// and a 512 px image from ~1 m/px imagery (z17), about 0.8 m per pixel.
	fineProfile = profile{
		level:    modelv1.Level_LEVEL_REFINED,
		gridStep: 10, satZoom: 17, imagePx: 512,
	}
	// coarseProfile is the background ring: a 40 m grid and a 64 px image
	// from ~8 m/px imagery (z14).
	coarseProfile = profile{
		level: modelv1.Level_LEVEL_PREVIEW, ring: true,
		gridStep: 40, satZoom: 14, imagePx: 64,
	}
)

type cellJob struct {
	x, y int
	dist float64
	prof *profile
	// What the cell still needs; set by the feed from what it already sent.
	needTerrain, needBuildings bool
}

// enuRect is an axis-aligned rectangle in local metres.
type enuRect struct {
	minE, minN, maxE, maxN float64
}

func (r enuRect) centre() (float64, float64) {
	return (r.minE + r.maxE) / 2, (r.minN + r.maxN) / 2
}

// expand scales the rectangle about its centre.
func (r enuRect) expand(f float64) enuRect {
	cE, cN := r.centre()
	hw := (r.maxE - r.minE) / 2 * f
	hh := (r.maxN - r.minN) / 2 * f
	return enuRect{cE - hw, cN - hh, cE + hw, cN + hh}
}

// cellRange is the inclusive index range of cells touching r.
func cellRange(r enuRect) (x0, y0, x1, y1 int) {
	x0 = int(math.Floor(r.minE / CellSizeM))
	y0 = int(math.Floor(r.minN / CellSizeM))
	x1 = max(x0, int(math.Ceil(r.maxE/CellSizeM))-1)
	y1 = max(y0, int(math.Ceil(r.maxN/CellSizeM))-1)
	return
}

// cellCentre is the ENU centre of cell (x, y).
func cellCentre(x, y int) (float64, float64) {
	return (float64(x) + 0.5) * CellSizeM, (float64(y) + 0.5) * CellSizeM
}

// planCells lists the cells of area (fine) and, when ring is set, of the
// background ring around it (coarse). Each group is ordered nearest to
// (cE, cN) first; the area always comes before the ring.
func planCells(area enuRect, cE, cN float64, ring bool) []cellJob {
	x0, y0, x1, y1 := cellRange(area)
	var core []cellJob
	inCore := func(x, y int) bool { return x >= x0 && x <= x1 && y >= y0 && y <= y1 }
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			core = append(core, newJob(x, y, cE, cN, &fineProfile))
		}
	}
	sortJobs(core)
	if len(core) > maxCoreCells {
		core = core[:maxCoreCells]
	}
	if !ring {
		return core
	}
	rx0, ry0, rx1, ry1 := cellRange(area.expand(RingSpan))
	var outer []cellJob
	for y := ry0; y <= ry1; y++ {
		for x := rx0; x <= rx1; x++ {
			if !inCore(x, y) {
				outer = append(outer, newJob(x, y, cE, cN, &coarseProfile))
			}
		}
	}
	sortJobs(outer)
	if len(outer) > maxRingCells {
		outer = outer[:maxRingCells]
	}
	return append(core, outer...)
}

func newJob(x, y int, cE, cN float64, p *profile) cellJob {
	e, n := cellCentre(x, y)
	return cellJob{x: x, y: y, dist: math.Hypot(e-cE, n-cN), prof: p}
}

// distTie is how close two cell distances must be to count as equal. Distances
// come through a GPS round trip, so exact ties differ in the last float bits.
const distTie = 1e-6

// sortJobs orders by distance, then row, then column, so ties are stable.
func sortJobs(js []cellJob) {
	sort.Slice(js, func(i, j int) bool {
		a, b := js[i], js[j]
		if math.Abs(a.dist-b.dist) > distTie {
			return a.dist < b.dist
		}
		if a.y != b.y {
			return a.y < b.y
		}
		return a.x < b.x
	})
}
