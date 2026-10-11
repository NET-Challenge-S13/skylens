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

// profile is the sampling density of one detail level.
type profile struct {
	level modelv1.Level
	// terrainStep is the terrain point spacing in metres.
	terrainStep float64
	// satZoom is the VWorld imagery zoom the colours come from.
	satZoom int
	// roofStep, wallStep and wallRise space building points in metres.
	roofStep, wallStep, wallRise float64
}

var (
	// fineProfile is the area itself: 2 m terrain, ~1 m/px imagery (z17).
	fineProfile = profile{
		level:       modelv1.Level_LEVEL_REFINED,
		terrainStep: 2, satZoom: 17,
		roofStep: 2, wallStep: 1.5, wallRise: 2,
	}
	// coarseProfile is the background ring: 8 m terrain, ~8 m/px imagery (z14).
	coarseProfile = profile{
		level:       modelv1.Level_LEVEL_PREVIEW,
		terrainStep: 8, satZoom: 14,
		roofStep: 5, wallStep: 5, wallRise: 4,
	}
)

type cellJob struct {
	x, y int
	dist float64
	prof *profile
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
