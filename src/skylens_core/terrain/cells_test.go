package terrain

import (
	"testing"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
)

func TestCellRangeCoversTheArea(t *testing.T) {
	x0, y0, x1, y1 := cellRange(enuRect{-300, -300, 300, 300})
	if x0 != -1 || y0 != -1 || x1 != 0 || y1 != 0 {
		t.Fatalf("range %d,%d..%d,%d, want -1,-1..0,0", x0, y0, x1, y1)
	}
	// An edge exactly on a cell boundary does not pull in the next cell.
	x0, y0, x1, y1 = cellRange(enuRect{0, 0, 800, 400})
	if x0 != 0 || y0 != 0 || x1 != 1 || y1 != 0 {
		t.Fatalf("range %d,%d..%d,%d, want 0,0..1,0", x0, y0, x1, y1)
	}
}

func TestPlanCellsNearestFirst(t *testing.T) {
	jobs := planCells(enuRect{-300, -300, 300, 300}, 200, 200, false)
	got := make([][2]int, len(jobs))
	for i, j := range jobs {
		got[i] = [2]int{j.x, j.y}
		if j.prof.level != modelv1.Level_LEVEL_REFINED {
			t.Errorf("cell %v level %v, want refined", got[i], j.prof.level)
		}
	}
	// (0,0) holds the centre; the two edge neighbours tie at 400 m and are
	// ordered by row; the diagonal comes last.
	want := [][2]int{{0, 0}, {0, -1}, {-1, 0}, {-1, -1}}
	if len(got) != len(want) {
		t.Fatalf("cells %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("cells %v, want %v", got, want)
		}
	}
}

func TestPlanCellsRingComesAfterTheAreaAtPreview(t *testing.T) {
	jobs := planCells(enuRect{-300, -300, 300, 300}, 0, 0, true)
	core := 0
	for i, j := range jobs {
		inside := j.x >= -1 && j.x <= 0 && j.y >= -1 && j.y <= 0
		if inside {
			core++
			if i >= 4 {
				t.Fatalf("area cell %d,%d at position %d, after the ring", j.x, j.y, i)
			}
			continue
		}
		if j.prof.level != modelv1.Level_LEVEL_PREVIEW {
			t.Errorf("ring cell %d,%d level %v, want preview", j.x, j.y, j.prof.level)
		}
		if i > 4 && jobs[i-1].dist > j.dist {
			t.Errorf("ring not nearest-first at %d", i)
		}
	}
	if core != 4 {
		t.Fatalf("%d area cells, want 4", core)
	}
	// Area x3 = -900..900 m -> cells -3..2 on both axes, 36 in all.
	if len(jobs) != 36 {
		t.Fatalf("%d cells, want 36", len(jobs))
	}
}
