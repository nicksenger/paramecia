//! Conway's Game of Life animation for loading screens.

use rand::Rng;
use ratatui::style::Color;

/// A Conway's Game of Life simulation.
pub struct GameOfLife {
    /// Current grid state (true = alive, false = dead).
    grid: Vec<Vec<bool>>,
    /// Grid width.
    width: usize,
    /// Grid height.
    height: usize,
    /// Generation counter.
    generation: usize,
}

impl GameOfLife {
    /// Create a new Game of Life with the given dimensions.
    /// Initializes with a random pattern seeded for interesting evolution.
    pub fn new(width: usize, height: usize) -> Self {
        let mut grid = vec![vec![false; width]; height];

        // Seed with some interesting patterns
        // Add a few gliders and random cells for visual interest
        let patterns = Self::initial_patterns(width, height);
        for (x, y) in patterns {
            if y < height && x < width {
                grid[y][x] = true;
            }
        }

        Self {
            grid,
            width,
            height,
            generation: 0,
        }
    }

    /// Generate initial patterns - mix of classic structures and random cells.
    /// Uses system entropy so each launch produces a different animation.
    fn initial_patterns(width: usize, height: usize) -> Vec<(usize, usize)> {
        let mut rng = rand::rng();
        let mut cells = Vec::new();

        // Place 2-4 gliders at random positions
        let glider_count = rng.random_range(2..=4);
        for _ in 0..glider_count {
            let ox = rng.random_range(0..width.saturating_sub(3));
            let oy = rng.random_range(0..height.saturating_sub(3));
            cells.push((ox + 1, oy));
            cells.push((ox + 2, oy + 1));
            cells.push((ox, oy + 2));
            cells.push((ox + 1, oy + 2));
            cells.push((ox + 2, oy + 2));
        }

        // Place an R-pentomino (chaotic pattern) at a random-ish center region
        let cx = rng.random_range(width / 4..width * 3 / 4);
        let cy = rng.random_range(height / 4..height * 3 / 4);
        cells.push((cx + 1, cy));
        cells.push((cx + 2, cy));
        cells.push((cx, cy + 1));
        cells.push((cx + 1, cy + 1));
        cells.push((cx + 1, cy + 2));

        // Optionally add a lightweight spaceship
        if width > 20 && height > 10 && rng.random_range(0..3) != 0 {
            let lx = rng.random_range(5..width.saturating_sub(6));
            let ly = rng.random_range(2..height.saturating_sub(5));
            cells.push((lx, ly + 1));
            cells.push((lx, ly + 3));
            cells.push((lx + 1, ly));
            cells.push((lx + 2, ly));
            cells.push((lx + 3, ly));
            cells.push((lx + 4, ly));
            cells.push((lx + 4, ly + 1));
            cells.push((lx + 4, ly + 2));
            cells.push((lx + 3, ly + 3));
        }

        // Scatter some random cells for additional variety
        let random_count = rng.random_range(10..30);
        for _ in 0..random_count {
            cells.push((rng.random_range(0..width), rng.random_range(0..height)));
        }

        cells
    }

    /// Advance the simulation by one generation.
    pub fn step(&mut self) {
        let mut new_grid = vec![vec![false; self.width]; self.height];

        for (y, new_row) in new_grid.iter_mut().enumerate() {
            for (x, cell) in new_row.iter_mut().enumerate() {
                let neighbors = self.count_neighbors(x, y);
                let alive = self.grid[y][x];

                // Conway's rules:
                // 1. Any live cell with 2 or 3 neighbors survives
                // 2. Any dead cell with exactly 3 neighbors becomes alive
                // 3. All other cells die or stay dead
                *cell = matches!((alive, neighbors), (true, 2) | (true, 3) | (false, 3));
            }
        }

        self.grid = new_grid;
        self.generation += 1;

        // If the grid becomes too empty, reseed
        let alive_count: usize = self
            .grid
            .iter()
            .map(|row| row.iter().filter(|&&c| c).count())
            .sum();
        if alive_count < 10 {
            self.reseed();
        }
    }

    /// Count live neighbors for a cell (with wrapping at edges).
    fn count_neighbors(&self, x: usize, y: usize) -> usize {
        let mut count = 0;

        for dy in [-1i32, 0, 1] {
            for dx in [-1i32, 0, 1] {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let nx = (x as i32 + dx).rem_euclid(self.width as i32) as usize;
                let ny = (y as i32 + dy).rem_euclid(self.height as i32) as usize;

                if self.grid[ny][nx] {
                    count += 1;
                }
            }
        }

        count
    }

    /// Reseed the grid with new patterns.
    fn reseed(&mut self) {
        self.grid = vec![vec![false; self.width]; self.height];
        let patterns = Self::initial_patterns(self.width, self.height);
        for (x, y) in patterns {
            if y < self.height && x < self.width {
                self.grid[y][x] = true;
            }
        }
    }

    /// Render the grid to a string with colored cells.
    /// Returns lines of (char, color_intensity) tuples for rendering.
    pub fn render(&self, color_index: usize, gradient: &[Color]) -> Vec<Vec<(char, Color)>> {
        let mut result = Vec::with_capacity(self.height);

        for y in 0..self.height {
            let mut row = Vec::with_capacity(self.width);
            for x in 0..self.width {
                if self.grid[y][x] {
                    // Alive cells get a color based on position and animation frame
                    let color_offset = (x + y + color_index) % gradient.len();
                    row.push(('█', gradient[color_offset]));
                } else {
                    // Dead cells are dimmed
                    row.push((' ', Color::Reset));
                }
            }
            result.push(row);
        }

        result
    }

    /// Get the current generation number.
    #[allow(dead_code)]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Get the current grid dimensions.
    #[allow(dead_code)]
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Resize the grid (used when terminal size changes).
    pub fn resize(&mut self, width: usize, height: usize) {
        if width != self.width || height != self.height {
            self.width = width;
            self.height = height;
            self.grid = vec![vec![false; width]; height];
            let patterns = Self::initial_patterns(width, height);
            for (x, y) in patterns {
                if y < height && x < width {
                    self.grid[y][x] = true;
                }
            }
        }
    }
}
