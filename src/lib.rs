//! Grid-based shortest path search (A*) with configurable neighborhoods,
//! traversal rules, wrapping behavior, edge costs and heuristics.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;

/// Trait for grid cell values so built-in rules can inspect them.
pub trait CellValue {
    /// Whether this cell blocks movement (used by [`no_one_allowed`]).
    fn is_blocked(&self) -> bool;
}

type NbhdFn<T, const N: usize, const M: usize> =
    fn((usize, usize), &[[T; M]; N]) -> Vec<(isize, isize)>;
type WrapFn = fn((isize, isize), (usize, usize)) -> (isize, isize);
type DistFn<T> = fn((usize, usize), (usize, usize), &T, &T) -> f64;
type HeuristicFn = fn((usize, usize), (usize, usize)) -> f64;

macro_rules! impl_cell_value_int {
    ($($t:ty),*) => {
        $(impl CellValue for $t {
            fn is_blocked(&self) -> bool {
                *self == 1
            }
        })*
    };
}

impl_cell_value_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

/// Configuration for the path search.
///
/// * `nbhd_fn` produces the raw candidate coordinates adjacent to a cell
///   (they may lie outside the grid; they are passed through `wrap` and
///   then checked against the grid bounds).
/// * `allowed` decides whether a cell may be entered.
/// * `wrap` maps an out-of-bounds candidate onto the torus (or leaves it
///   untouched, in which case it is simply rejected).
/// * `dist` is the cost of stepping from one cell to another.
/// * `heuristic` is the A* lower-bound estimate from a cell to the goal.
pub struct Options<T, const N: usize, const M: usize> {
    pub nbhd_fn: NbhdFn<T, N, M>,
    pub allowed: fn(&T) -> bool,
    pub wrap: WrapFn,
    pub dist: DistFn<T>,
    pub heuristic: HeuristicFn,
}

impl<T: CellValue, const N: usize, const M: usize> Default for Options<T, N, M> {
    /// Moore neighborhood, cells equal to 1 blocked, no wrapping,
    /// Euclidean step costs and the Euclidean-distance heuristic.
    fn default() -> Self {
        Options {
            nbhd_fn: moore_nbhd,
            allowed: |v| !v.is_blocked(),
            wrap: no_wrap,
            dist: euclidean_step,
            heuristic: euclidean_heuristic,
        }
    }
}

/// The result of a search. On failure `path` is empty and `length` is `None`.
#[derive(Debug, PartialEq)]
pub struct Path {
    pub path: Vec<(usize, usize)>,
    pub length: Option<f64>,
}

// ---------------------------------------------------------------------------
// Built-in configuration pieces
// ---------------------------------------------------------------------------

/// Blocks exactly those cells whose value equals 1.
pub fn no_one_allowed<T: CellValue>(val: &T) -> bool {
    !val.is_blocked()
}

/// Identity mapping: out-of-bounds coordinates stay out of bounds.
pub fn no_wrap(i1: (isize, isize), _dims: (usize, usize)) -> (isize, isize) {
    i1
}

/// Toroidal wrapping modulo the grid dimensions.
pub fn wrap(i1: (isize, isize), dims: (usize, usize)) -> (isize, isize) {
    (
        i1.0.rem_euclid(dims.0 as isize),
        i1.1.rem_euclid(dims.1 as isize),
    )
}

/// Every step costs 1 (including diagonals).
pub fn unit_dist<T>(_i1: (usize, usize), _i2: (usize, usize), _v1: &T, _v2: &T) -> f64 {
    1.0
}

/// Step cost equal to the Euclidean distance between the two cells,
/// so diagonal moves cost sqrt(2). Pairs with [`euclidean_heuristic`].
pub fn euclidean_step<T>(i1: (usize, usize), i2: (usize, usize), _v1: &T, _v2: &T) -> f64 {
    euclidean_heuristic(i1, i2)
}

/// Euclidean distance between two cells (used as the default heuristic).
pub fn euclidean_heuristic(i1: (usize, usize), i2: (usize, usize)) -> f64 {
    let dr = i1.0 as f64 - i2.0 as f64;
    let dc = i1.1 as f64 - i2.1 as f64;
    (dr * dr + dc * dc).sqrt()
}

fn in_bounds(c: (isize, isize), dims: (usize, usize)) -> bool {
    c.0 >= 0 && c.1 >= 0 && (c.0 as usize) < dims.0 && (c.1 as usize) < dims.1
}

/// All 8 Moore neighbors (diagonals included).
pub fn moore_nbhd<T, const N: usize, const M: usize>(
    cell: (usize, usize),
    _grid: &[[T; M]; N],
) -> Vec<(isize, isize)> {
    let (r, c) = (cell.0 as isize, cell.1 as isize);
    let mut out = Vec::with_capacity(8);
    for dr in -1..=1isize {
        for dc in -1..=1isize {
            if dr != 0 || dc != 0 {
                out.push((r + dr, c + dc));
            }
        }
    }
    out
}

/// The 4 von Neumann neighbors (no diagonals).
pub fn von_neumann_nbhd<T, const N: usize, const M: usize>(
    cell: (usize, usize),
    _grid: &[[T; M]; N],
) -> Vec<(isize, isize)> {
    let (r, c) = (cell.0 as isize, cell.1 as isize);
    vec![(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
}

// ---------------------------------------------------------------------------
// A* search
// ---------------------------------------------------------------------------

/// Find the shortest path across `grid` from `start` to `goal` using A*.
///
/// Returns a `Path` whose `path` includes both endpoints; if no route
/// exists the `path` is empty and `length` is `None`.
pub fn shortest_path<T, const N: usize, const M: usize>(
    grid: &[[T; M]; N],
    config: &Options<T, N, M>,
    start: (usize, usize),
    goal: (usize, usize),
) -> Path {
    let dims = (N, M);
    let idx = |c: (usize, usize)| c.0 * M + c.1;

    if !in_bounds((start.0 as isize, start.1 as isize), dims)
        || !in_bounds((goal.0 as isize, goal.1 as isize), dims)
        || !(config.allowed)(&grid[start.0][start.1])
        || !(config.allowed)(&grid[goal.0][goal.1])
    {
        return Path {
            path: Vec::new(),
            length: None,
        };
    }

    let mut g_score = vec![f64::INFINITY; N * M];
    let mut came_from: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut closed = vec![false; N * M];

    g_score[idx(start)] = 0.0;

    // Max-heap keyed by Reverse(bits): for non-negative floats, bit
    // patterns are ordered like the values, so this pops the smallest
    // f-score first.
    let mut open: BinaryHeap<(Reverse<u64>, usize)> = BinaryHeap::new();
    open.push((
        Reverse((config.heuristic)(start, goal).to_bits()),
        idx(start),
    ));

    while let Some((Reverse(_key), cur_idx)) = open.pop() {
        if closed[cur_idx] {
            continue; // stale heap entry
        }
        closed[cur_idx] = true;

        let cur = (cur_idx / M, cur_idx % M);
        if cur == goal {
            let mut path = vec![goal];
            let mut node = goal;
            while let Some(&prev) = came_from.get(&idx(node)) {
                path.push(prev);
                node = prev;
            }
            path.reverse();
            return Path {
                path,
                length: Some(g_score[idx(goal)]),
            };
        }

        for raw in (config.nbhd_fn)(cur, grid) {
            let cand = (config.wrap)(raw, dims);
            if !in_bounds(cand, dims) {
                continue;
            }
            let next = (cand.0 as usize, cand.1 as usize);
            let n_idx = idx(next);
            if closed[n_idx] || !(config.allowed)(&grid[next.0][next.1]) {
                continue;
            }
            let step =
                (config.dist)(cur, next, &grid[cur.0][cur.1], &grid[next.0][next.1]);
            let tentative = g_score[cur_idx] + step;
            if tentative < g_score[n_idx] {
                g_score[n_idx] = tentative;
                came_from.insert(n_idx, cur);
                let f = tentative + (config.heuristic)(next, goal);
                open.push((Reverse(f.to_bits()), n_idx));
            }
        }
    }

    Path {
        path: Vec::new(),
        length: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_line_moore() {
        let grid = [[0u8; 3]; 3];
        let config = Options::default();
        let p = shortest_path(&grid, &config, (0, 0), (2, 2));
        assert_eq!(p.length, Some(8f64.sqrt()));
        assert_eq!(p.path.first(), Some(&(0, 0)));
        assert_eq!(p.path.last(), Some(&(2, 2)));
    }

    #[test]
    fn wall_forces_detour() {
        let grid = [
            [0u8, 1, 0],
            [0u8, 1, 0],
            [0u8, 0, 0],
        ];
        let config = Options::default();
        let p = shortest_path(&grid, &config, (0, 0), (0, 2));
        assert_eq!(
            p.path,
            vec![(0, 0), (1, 0), (2, 1), (1, 2), (0, 2)]
        );
        let expected = 2.0 + 2.0 * 2f64.sqrt();
        assert_eq!(p.length, Some(expected));
    }

    #[test]
    fn von_neumann_is_longer_than_moore() {
        let grid = [[0u8; 3]; 3];
        let config = Options::<u8, 3, 3> {
            nbhd_fn: von_neumann_nbhd,
            ..Options::default()
        };
        let p = shortest_path(&grid, &config, (0, 0), (2, 2));
        assert_eq!(p.length, Some(4.0));
    }

    #[test]
    fn no_path_when_fully_walled() {
        let grid = [
            [0u8, 1, 0],
            [1u8, 1, 0],
            [0u8, 0, 0],
        ];
        let config = Options::default();
        let p = shortest_path(&grid, &config, (0, 0), (0, 2));
        assert_eq!(p, Path { path: Vec::new(), length: None });
    }

    #[test]
    fn wrapping_shortcuts_the_grid() {
        let grid = [[0u8; 3]; 3];
        let config = Options::<u8, 3, 3> {
            wrap,
            dist: unit_dist,
            ..Options::default()
        };
        // With a torus, moving one row up from (0,1) lands on (2,1).
        let p = shortest_path(&grid, &config, (0, 1), (2, 1));
        assert_eq!(p.path, vec![(0, 1), (2, 1)]);
        assert_eq!(p.length, Some(1.0));
    }

    #[test]
    fn custom_heuristic_and_dist_are_honored() {
        // Zero heuristic turns A* into Dijkstra; weighted edges via dist.
        // Von Neumann nbhd so there is no diagonal shortcut.
        let grid = [[0u8; 2]; 2];
        let config = Options::<u8, 2, 2> {
            nbhd_fn: von_neumann_nbhd,
            dist: |a, b, _, _| if a.0 != b.0 { 10.0 } else { 1.0 },
            heuristic: |_, _| 0.0,
            ..Options::default()
        };
        let p = shortest_path(&grid, &config, (0, 0), (1, 1));
        // Cheapest route avoids row changes: (0,0) -> (0,1) -> (1,1) costs 11.
        assert_eq!(p.length, Some(11.0));
    }

    #[test]
    fn start_equals_goal() {
        let grid = [[0u8; 2]; 2];
        let config = Options::default();
        let p = shortest_path(&grid, &config, (1, 1), (1, 1));
        assert_eq!(p.path, vec![(1, 1)]);
        assert_eq!(p.length, Some(0.0));
    }

    #[test]
    fn blocked_start_or_goal_fails() {
        let grid = [[1u8, 0], [0, 0]];
        let config = Options::default();
        let p = shortest_path(&grid, &config, (0, 0), (1, 1));
        assert_eq!(p, Path { path: Vec::new(), length: None });
    }
}
