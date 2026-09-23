//! A survival-first Battlesnake with local geometry and lightweight adversarial search.
//!
//! The priority order is: avoid certain death, avoid losing head-to-heads, avoid
//! traps, preserve future manoeuvring room, account for enemy pressure, then use
//! territory and food to improve the position. The strategic search is bounded
//! and deterministic so bad decisions remain reproducible from replays.

use log::{debug, info, warn};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::TryFrom;
use std::time::Instant;

use crate::{Battlesnake, Board, Coord, Game};

const DIRECTIONS: [Direction; 4] = [
    Direction::Up,
    Direction::Left,
    Direction::Right,
    Direction::Down,
];

const SPACE_WEIGHT: i64 = 140;
const EXIT_WEIGHT: i64 = 90;
const SPACE_DEFICIT_WEIGHT: i64 = 900;
const HEALTH_WEIGHT: i64 = 3;
const HAZARD_WEIGHT: i64 = 120;
const FOOD_HEALTH_THRESHOLD: i32 = 45;
const FOOD_HEALTH_WEIGHT: i64 = 95;
const FOOD_SCORE: i64 = -250;
const FOOD_SEEK_HEALTH: i32 = 65;
const FOOD_URGENCY_HEALTH: i32 = 55;
const FOOD_DISTANCE_WEIGHT: i64 = 25;
const CRITICAL_HEALTH: i32 = 18;
const STARVATION_PENALTY: i64 = 4_000;
const NO_FOOD_PENALTY: i64 = 5_000;
const SURVIVAL_HORIZON: usize = 20;
const SURVIVAL_BEAM_WIDTH: usize = 64;
const TERRITORY_WEIGHT: i64 = 90;
const FORCED_KILL_BONUS: i64 = 12_000;
const MIN_SAFE_SPACE: usize = 8;
const MIN_SAFE_FUTURE_SURVIVAL: i64 = 4_000;
const MIN_SAFE_EXITS: i64 = 2;
const TRAP_SPACE_THRESHOLD: i64 = 24;
const TRAP_PENALTY_WEIGHT: i64 = 320;

// V2 strategic weights. These sit below the hard survival filters.
const TRAP_RISK_WEIGHT: i64 = 12;
const ENEMY_PRESSURE_WEIGHT: i64 = 5;
const ADVERSARIAL_SPACE_WEIGHT: i64 = 0;
const ADVERSARIAL_COLLAPSE_WEIGHT: i64 = 700;
const FUTURE_SPACE_WEIGHT: i64 = 14;
const MIN_ADVERSARIAL_SPACE: usize = 10;
const PRESSURE_DISTANCE: i32 = 5;
const TRAP_LOOKAHEAD: i32 = 6;
const MIN_PREFERRED_EXITS: i64 = 3;
const MIN_PREFERRED_FUTURE_SPACE: i64 = 100;

// V2.2 anti-corner / anti-push analysis. These values are intentionally
// strong: losing our last escape route is much worse than giving up some
// territory or a nearby food pickup.
const WORST_CASE_SPACE_WEIGHT: i64 = 28;
const WORST_CASE_EXIT_WEIGHT: i64 = 1_150;
const ESCAPE_ROUTE_WEIGHT: i64 = 240;
const ENEMY_CUTOFF_WEIGHT: i64 = 18;
const ENEMY_PUSH_WEIGHT: i64 = 14;
const EDGE_EXPOSURE_WEIGHT: i64 = 11;
const PUSH_DISTANCE_THRESHOLD: i64 = 8;

// V2.3 strategic anti-cornering. We now look one enemy response and one
// defensive move further: a position is considered strategically unsafe when
// the opponent can force our best continuation into a one-exit state or keep
// us moving toward the boundary.
const MIN_STRATEGIC_ESCAPE_ROUTES: i64 = 2;
const ENEMY_PUSH_DANGER: i64 = 1_800;
const ENEMY_CUTOFF_DANGER: i64 = 2_000;
const EDGE_DANGER: i64 = 1_000;
const STRATEGIC_COMMITMENT_WEIGHT: i64 = 520;
const STRATEGIC_ESCAPE_WEIGHT: i64 = 420;
const FOOD_IN_CERCO_PENALTY: i64 = 18_000;

const FOOD_RESCUE_HEALTH: i32 = 35;
const FOOD_RESCUE_WEIGHT: i64 = 180;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Point {
    x: i32,
    y: i32,
}

impl From<&Coord> for Point {
    fn from(coord: &Coord) -> Self {
        Self {
            x: coord.x,
            y: coord.y,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    fn name(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    fn next(self, from: Point, board: &Board, wraps: bool) -> Option<Point> {
        let (dx, dy) = match self {
            Self::Up => (0, 1),
            Self::Down => (0, -1),
            Self::Left => (-1, 0),
            Self::Right => (1, 0),
        };
        let next = Point {
            x: from.x + dx,
            y: from.y + dy,
        };

        if wraps {
            // A zero-sized board is invalid according to the API, but avoid a
            // panic if a malformed local fixture is ever sent to the server.
            if board.width <= 0 || board.height == 0 {
                return None;
            }
            return Some(Point {
                x: next.x.rem_euclid(board.width),
                y: next.y.rem_euclid(board.height),
            });
        }

        in_bounds(next, board).then_some(next)
    }
}

#[derive(Debug)]
struct Candidate {
    direction: Direction,
    score: i64,
    health_after: i32,
    space: usize,
    exits: i64,
    eating: bool,
    territory: i64,
    forced_kill: bool,
    future_survival: i64,
    trap_risk: i64,
    enemy_pressure: i64,
    adversarial_space: usize,
    future_space: i64,
    worst_case_space: usize,
    worst_case_exits: i64,
    escape_routes: i64,
    future_worst_exits: i64,
    future_escape_routes: i64,
    enemy_cutoff_risk: i64,
    enemy_push_risk: i64,
    edge_exposure_risk: i64,
    commitment_risk: i64,
    loses_head_to_head: bool,
}

struct TerritoryAnalysis {
    controlled: i64,
    enemy_distances: Vec<(i32, HashMap<Point, i32>)>,
}

#[derive(Clone)]
struct SimState {
    body: Vec<Point>,
    health: i32,
    eaten_food: u128,
}

struct SurvivalContext<'a> {
    game: &'a Game,
    board: &'a Board,
    you: &'a Battlesnake,
    hazard_stacks: &'a HashMap<Point, i32>,
    dangerous: &'a HashSet<Point>,
    enemy_arrivals: &'a [(i32, HashMap<Point, i32>)],
    wraps: bool,
}

struct AttackContext<'a> {
    game: &'a Game,
    board: &'a Board,
    you: &'a Battlesnake,
    projected_body: &'a [Point],
    food: &'a HashSet<Point>,
    wraps: bool,
}

// info is called when you create your Battlesnake on play.battlesnake.com
// and controls your Battlesnake's appearance.
pub fn info() -> Value {
    info!("INFO");

    json!({
        "apiversion": "1",
        "author": "",
        "color": "#276FBF",
        "head": "default",
        "tail": "default",
    })
}

pub fn start(game: &Game, turn: &i32, board: &Board, you: &Battlesnake) {
    info!(
        "GAME_START id={} turn={} board={}x{} snakes={} health={} length={}",
        game.id,
        turn,
        board.width,
        board.height,
        board.snakes.len(),
        you.health,
        you.length
    );
}

pub fn end(game: &Game, turn: &i32, board: &Board, you: &Battlesnake) {
    info!(
        "GAME_END id={} turn={} head=({}, {}) health={} length={} survivors={}",
        game.id,
        turn,
        you.head.x,
        you.head.y,
        you.health,
        you.length,
        board.snakes.len()
    );
}

/// Select the highest-scoring survivable move.
///
/// `Candidate` only contains moves that do not immediately leave the board,
/// hit a body, or run out of health. A losing head-to-head is retained as a
/// last resort, but is never selected when another survivable move exists.
fn select_direction(game: &Game, board: &Board, you: &Battlesnake) -> Direction {
    let candidates: Vec<Candidate> = DIRECTIONS
        .iter()
        .filter_map(|direction| evaluate_move(*direction, game, board, you))
        .collect();

    if candidates.is_empty() {
        // The game is already lost. Return a syntactically valid direction
        // rather than timing out or panicking; an in-bounds move is preferred
        // for useful diagnostics in the replay.
        warn!(
            "MOVE_NO_SAFE_CANDIDATE game_id={} ruleset={} health={} head=({}, {})",
            game.id,
            game.ruleset
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            you.health,
            you.head.x,
            you.head.y
        );
        let fallback = DIRECTIONS
            .iter()
            .copied()
            .find(|direction| {
                direction
                    .next(Point::from(&you.head), board, is_wrapped(game))
                    .is_some()
            })
            .unwrap_or(Direction::Up);
        warn!("MOVE_FALLBACK direction={}", fallback.name());
        return fallback;
    }

    let non_losing: Vec<&Candidate> = candidates
        .iter()
        .filter(|candidate| !candidate.loses_head_to_head)
        .collect();
    let pool: Vec<&Candidate> = if non_losing.is_empty() {
        candidates.iter().collect()
    } else {
        non_losing
    };

    // Do not choose a move whose continuation search is already dead when
    // another candidate can continue. This is stronger than a score penalty:
    // food or territory must not beat a route with a real future.
    let future_viable: Vec<&Candidate> = pool
        .iter()
        .copied()
        .filter(|candidate| candidate.future_survival > 0)
        .collect();
    let pool = if future_viable.is_empty() {
        pool
    } else {
        future_viable
    };

    // Do not trade a healthy escape route for a locally attractive corridor.
    // Keep the mobility floor as a preference, not an absolute rule: when
    // every legal move is already dangerous, the normal score still chooses
    // the least-bad option and the logs preserve that situation.
    let mobility_safe: Vec<&Candidate> = pool
        .iter()
        .copied()
        .filter(|candidate| {
            candidate.space >= MIN_SAFE_SPACE
                && candidate.future_survival >= MIN_SAFE_FUTURE_SURVIVAL
                && candidate.exits >= MIN_SAFE_EXITS
        })
        .collect();
    let pool = if mobility_safe.is_empty() {
        let best_future = pool
            .iter()
            .map(|candidate| candidate.future_survival)
            .max()
            .unwrap_or(0);
        let future_best: Vec<&Candidate> = pool
            .iter()
            .copied()
            .filter(|candidate| candidate.future_survival == best_future)
            .collect();
        if future_best.is_empty() {
            pool
        } else {
            future_best
        }
    } else {
        mobility_safe
    };

    // Prefer genuinely open positions when one exists. This prevents the
    // snake from repeatedly choosing 2-exit edge/corridor moves merely because
    // their current territory score is slightly higher. If no such move exists,
    // fall back to the normal pool.
    let open_positions: Vec<&Candidate> = pool
        .iter()
        .copied()
        .filter(|candidate| {
            candidate.exits >= MIN_PREFERRED_EXITS
                && candidate.future_space >= MIN_PREFERRED_FUTURE_SPACE
        })
        .collect();
    let pool = if open_positions.is_empty() {
        pool
    } else {
        open_positions
    };

    // V2.3: strategic escape preservation. We first prefer moves for which
    // the enemy's best response still leaves us a real defensive continuation.
    // This is deliberately a preference with a fallback: on genuinely bad
    // boards we still choose the least-bad legal move instead of panicking.
    let strategic_safe: Vec<&Candidate> = pool
        .iter()
        .copied()
        .filter(|candidate| candidate.forced_kill || !is_cerco_risk(candidate))
        .collect();
    let pool = if strategic_safe.is_empty() {
        pool
    } else {
        strategic_safe
    };

    finish_selection(game, pool)

}

fn finish_selection(game: &Game, pool: Vec<&Candidate>) -> Direction {
    // `>` intentionally preserves DIRECTIONS' stable tie-break order.
    let mut best = pool[0];
    for candidate in pool.into_iter().skip(1) {
        if candidate.score > best.score {
            best = candidate;
        }
    }
    info!(
        "MOVE_DECISION id={} direction={} score={} health_after={} space={} territory={} exits={} eating={} forced_kill={} h2h_risk={} future={} trap={} pressure={} adversarial_space={} future_space={} worst_space={} worst_exits={} escape_routes={} future_worst_exits={} future_escape_routes={} cutoff={} push={} edge_risk={} commitment={}",
        game.id,
        best.direction.name(),
        best.score,
        best.health_after,
        best.space,
        best.territory,
        best.exits,
        best.eating,
        best.forced_kill,
        best.loses_head_to_head,
        best.future_survival,
        best.trap_risk,
        best.enemy_pressure,
        best.adversarial_space,
        best.future_space,
        best.worst_case_space,
        best.worst_case_exits,
        best.escape_routes,
        best.future_worst_exits,
        best.future_escape_routes,
        best.enemy_cutoff_risk,
        best.enemy_push_risk,
        best.edge_exposure_risk,
        best.commitment_risk
    );
    best.direction
}

fn evaluate_move(
    direction: Direction,
    game: &Game,
    board: &Board,
    you: &Battlesnake,
) -> Option<Candidate> {
    let wraps = is_wrapped(game);
    let head = Point::from(&you.head);
    let Some(target) = direction.next(head, board, wraps) else {
        debug!(
            "MOVE_REJECT direction={} reason=outside_board head=({}, {}) wraps={}",
            direction.name(),
            head.x,
            head.y,
            wraps
        );
        return None;
    };
    let food = point_set(&board.food);
    let eating = food.contains(&target);
    let constrictor = is_constrictor(game);
    let grows = eating || constrictor;
    let my_length = snake_length(you) + i32::from(grows);
    let hazard_damage = hazard_damage(game);
    let hazard_stacks = hazard_counts(&board.hazards);
    let hazard_cost = if eating || constrictor {
        // Food restores health in the same turn, including when it is in a
        // hazard. Constrictor does not lose health between turns.
        0
    } else {
        hazard_stacks.get(&target).copied().unwrap_or(0) * hazard_damage
    };
    let health_after = if eating || constrictor {
        100
    } else {
        you.health - 1 - hazard_cost
    };

    if health_after <= 0 {
        debug!(
            "MOVE_REJECT direction={} reason=lethal_health target=({}, {}) health_after={} hazard_cost={} eating={}",
            direction.name(),
            target.x,
            target.y,
            health_after,
            hazard_cost,
            eating
        );
        return None;
    }
    if collides_with_body(target, board, you, grows) {
        debug!(
            "MOVE_REJECT direction={} reason=body_collision target=({}, {}) grows={}",
            direction.name(),
            target.x,
            target.y,
            grows
        );
        return None;
    }

    let loses_head_to_head = board
        .snakes
        .iter()
        .filter(|snake| snake.id != you.id)
        .any(|snake| {
            can_reach_in_one_turn(snake, target, board, wraps)
                && snake_length(snake) + i32::from(eating) >= my_length
        });

    let projected_body = projected_body(you, target, grows);
    let dangerous = larger_head_territory(board, you, my_length, wraps);
    let (blocked, traversable_tail) = blocked_after_move(board, you, &projected_body, grows);
    let mut planning_blocked = blocked;

    // Hazards are a poor long-term escape route. We permit landing on food in
    // a hazard (which is safe), but otherwise do not count hazardous cells as
    // future room. Immediate hazard moves are still considered above when
    // health permits them, so this is conservative rather than impossible.
    for hazard in hazard_stacks.keys() {
        if !food.contains(hazard) && *hazard != target {
            planning_blocked.insert(*hazard);
        }
    }
    if traversable_tail {
        if let Some(tail) = you.body.last().map(Point::from) {
            planning_blocked.remove(&tail);
        }
    }
    let (space, distances) = reachable_space(board, target, &planning_blocked, &dangerous, wraps);
    let exits = DIRECTIONS
        .iter()
        .filter_map(|next_direction| next_direction.next(target, board, wraps))
        .filter(|point| !planning_blocked.contains(point) && !dangerous.contains(point))
        .count() as i64;
    let territory = territory_analysis(board, you, target, &projected_body, my_length, wraps);
    let (food_distance, food_is_contested) = nearest_food(
        &food,
        target,
        my_length,
        &distances,
        &territory.enemy_distances,
    );
    let survival_context = SurvivalContext {
        game,
        board,
        you,
        hazard_stacks: &hazard_stacks,
        dangerous: &dangerous,
        enemy_arrivals: &territory.enemy_distances,
        wraps,
    };
    let future_survival = future_survival_score(
        &survival_context,
        &projected_body,
        health_after,
        target,
        eating,
    );

    // A move can have plenty of space now while putting us inside a corridor
    // that collapses a few turns later.
    let trap_risk = trap_risk_score(
        board,
        target,
        &planning_blocked,
        &dangerous,
        wraps,
        projected_body.len(),
    );

    let enemy_pressure = enemy_pressure_score(
        board,
        you,
        target,
        wraps,
    );

    // V2.2: simulate one plausible response from each enemy and measure how
    // much of our mobility remains. This is the missing layer for enemies that
    // intentionally push us toward a wall/corner.
    let adversarial = adversarial_escape_analysis(
        game,
        board,
        you,
        target,
        &projected_body,
        &dangerous,
        &hazard_stacks,
        wraps,
        exits,
        space,
        health_after,
    );
    let adversarial_space = adversarial.worst_case_space;

    let future_space = future_space_score(
        board,
        target,
        &planning_blocked,
        &dangerous,
        wraps,
        projected_body.len(),
    );

    let attack_context = AttackContext {
        game,
        board,
        you,
        projected_body: &projected_body,
        food: &food,
        wraps,
    };
    let forced_kill = forced_kill_available(&attack_context, target, my_length);

    let projected_length = projected_body.len() as i64;
    let space_deficit = (projected_length + 2 - space as i64).max(0);
    let mut score = space as i64 * SPACE_WEIGHT + exits * EXIT_WEIGHT + centre_score(target, board);
    score += health_after as i64 * HEALTH_WEIGHT;
    score -= hazard_cost as i64 * HAZARD_WEIGHT;
    score -= space_deficit * space_deficit * SPACE_DEFICIT_WEIGHT;
    score += territory.controlled * TERRITORY_WEIGHT;
    score += future_survival;

    score -= trap_risk * TRAP_RISK_WEIGHT;
    score -= enemy_pressure * ENEMY_PRESSURE_WEIGHT;
    score += adversarial_space as i64 * ADVERSARIAL_SPACE_WEIGHT;
    score += future_space * FUTURE_SPACE_WEIGHT;

    // V2.2 anti-corner scoring. A move is only truly good if it keeps an
    // escape after the opponent gets the next move.
    score += adversarial.worst_case_space as i64 * WORST_CASE_SPACE_WEIGHT;
    score += adversarial.worst_case_exits * WORST_CASE_EXIT_WEIGHT;
    score += adversarial.escape_routes * ESCAPE_ROUTE_WEIGHT;
    score -= adversarial.enemy_cutoff_risk * ENEMY_CUTOFF_WEIGHT;
    score -= adversarial.enemy_push_risk * ENEMY_PUSH_WEIGHT;
    score -= adversarial.edge_exposure_risk * EDGE_EXPOSURE_WEIGHT;

    // V2.3: reward a stable escape chain, not just a single good square.
    score += adversarial.future_worst_exits * STRATEGIC_ESCAPE_WEIGHT;
    score += adversarial.future_escape_routes * STRATEGIC_ESCAPE_WEIGHT;
    score -= adversarial.commitment_risk * STRATEGIC_COMMITMENT_WEIGHT;

    // Optional food is deliberately unattractive inside an active enemy
    // funnel. Rescue food still receives the normal low-health treatment below.
    if eating && you.health > FOOD_HEALTH_THRESHOLD {
        if is_cerco_risk_values(&adversarial) {
            score -= FOOD_IN_CERCO_PENALTY;
        }
    }

    if adversarial_space < MIN_ADVERSARIAL_SPACE {
        let deficit = (MIN_ADVERSARIAL_SPACE - adversarial_space) as i64;
        score -= deficit * deficit * ADVERSARIAL_COLLAPSE_WEIGHT;
    }

    if exits < MIN_SAFE_EXITS && space < TRAP_SPACE_THRESHOLD as usize {
        let deficit = TRAP_SPACE_THRESHOLD - space as i64;
        score -= deficit * deficit * TRAP_PENALTY_WEIGHT;
    }
    if forced_kill {
        score += FORCED_KILL_BONUS;
    }

    if eating {
        // Food is mainly valuable as fuel; growing without a health need is a
        // modest cost because it reduces future manoeuvrability.
        score += if you.health <= FOOD_HEALTH_THRESHOLD {
            (101 - you.health) as i64 * FOOD_HEALTH_WEIGHT
        } else if you.health <= FOOD_SEEK_HEALTH {
            (FOOD_SEEK_HEALTH - you.health + 1) as i64 * FOOD_HEALTH_WEIGHT
        } else {
            FOOD_SCORE
        };
    } else if health_after <= FOOD_SEEK_HEALTH {
        if let Some(distance) = food_distance {
            let can_arrive_before_starving = constrictor || distance <= health_after;
            if can_arrive_before_starving && !food_is_contested {
                let urgency = (FOOD_URGENCY_HEALTH - health_after).max(0) as i64;
                score += urgency * FOOD_HEALTH_WEIGHT - distance as i64 * FOOD_DISTANCE_WEIGHT;
            } else if health_after <= FOOD_RESCUE_HEALTH && distance <= health_after {
                score += (FOOD_RESCUE_HEALTH - health_after + 1) as i64 * FOOD_RESCUE_WEIGHT
                    - distance as i64 * FOOD_DISTANCE_WEIGHT;
            } else if health_after <= CRITICAL_HEALTH {
                score -= STARVATION_PENALTY;
            }
        } else if !constrictor && health_after <= CRITICAL_HEALTH {
            score -= NO_FOOD_PENALTY;
        }
    }

    let candidate = Candidate {
        direction,
        score,
        health_after,
        space,
        exits,
        eating,
        territory: territory.controlled,
        forced_kill,
        future_survival,
        trap_risk,
        enemy_pressure,
        adversarial_space,
        future_space,
        worst_case_space: adversarial.worst_case_space,
        worst_case_exits: adversarial.worst_case_exits,
        escape_routes: adversarial.escape_routes,
        future_worst_exits: adversarial.future_worst_exits,
        future_escape_routes: adversarial.future_escape_routes,
        enemy_cutoff_risk: adversarial.enemy_cutoff_risk,
        enemy_push_risk: adversarial.enemy_push_risk,
        edge_exposure_risk: adversarial.edge_exposure_risk,
        commitment_risk: adversarial.commitment_risk,
        loses_head_to_head,
    };
    debug!("MOVE_CANDIDATE {:?}", candidate);
    Some(candidate)
}

fn in_bounds(point: Point, board: &Board) -> bool {
    point.x >= 0 && point.x < board.width && point.y >= 0 && point.y < board.height
}

fn point_set(coords: &[Coord]) -> HashSet<Point> {
    coords.iter().map(Point::from).collect()
}

fn hazard_counts(hazards: &[Coord]) -> HashMap<Point, i32> {
    let mut counts = HashMap::new();
    for hazard in hazards {
        *counts.entry(Point::from(hazard)).or_insert(0) += 1;
    }
    counts
}

fn snake_length(snake: &Battlesnake) -> i32 {
    snake.length.max(snake.body.len() as i32)
}

fn projected_body(you: &Battlesnake, target: Point, grows: bool) -> Vec<Point> {
    let mut result = Vec::with_capacity(you.body.len() + 1);
    result.push(target);
    result.extend(you.body.iter().map(Point::from));
    if !grows {
        result.pop();
    }
    result
}

fn collides_with_body(target: Point, board: &Board, you: &Battlesnake, grows: bool) -> bool {
    for snake in &board.snakes {
        if snake.id == you.id {
            let body_end = if grows {
                snake.body.len()
            } else {
                snake.body.len().saturating_sub(1)
            };
            if snake.body[..body_end]
                .iter()
                .any(|segment| Point::from(segment) == target)
            {
                return true;
            }
        } else if snake
            .body
            .iter()
            .any(|segment| Point::from(segment) == target)
        {
            // Other tails are intentionally blocked. An opponent can eat and
            // keep its tail in place, so treating it as always free is fatal.
            return true;
        }
    }
    false
}

fn blocked_after_move(
    board: &Board,
    you: &Battlesnake,
    projected_body: &[Point],
    grows: bool,
) -> (HashSet<Point>, bool) {
    let mut blocked = HashSet::new();
    for snake in &board.snakes {
        if snake.id != you.id {
            blocked.extend(snake.body.iter().map(Point::from));
        }
    }
    blocked.extend(projected_body.iter().copied());
    // A non-growing tail will be available on the following turn. Counting it
    // as reachable avoids rejecting safe tail-chasing corridors.
    (blocked, !grows)
}

fn can_reach_in_one_turn(snake: &Battlesnake, target: Point, board: &Board, wraps: bool) -> bool {
    let head = Point::from(&snake.head);
    let neck = snake.body.get(1).map(Point::from);
    DIRECTIONS.iter().copied().any(|direction| {
        let next = direction.next(head, board, wraps);
        next == Some(target) && neck != Some(target)
    })
}

fn larger_head_territory(
    board: &Board,
    you: &Battlesnake,
    my_length: i32,
    wraps: bool,
) -> HashSet<Point> {
    let mut territory = HashSet::new();
    for snake in board.snakes.iter().filter(|snake| snake.id != you.id) {
        if snake_length(snake) < my_length {
            continue;
        }
        let head = Point::from(&snake.head);
        let neck = snake.body.get(1).map(Point::from);
        for direction in DIRECTIONS.iter().copied() {
            if let Some(point) = direction.next(head, board, wraps) {
                if neck != Some(point) {
                    territory.insert(point);
                }
            }
        }
    }
    territory
}

/// Partition the currently accessible board by exact path distance.
///
/// A normal flood fill says that every connected free cell is "our" room. In
/// multiplayer that is false: a rival head can arrive first and turn the same
/// cell into a losing head-to-head. This Voronoi-style pass only credits cells
/// we reach sooner, plus ties that our longer body can win.
fn territory_analysis(
    board: &Board,
    you: &Battlesnake,
    my_target: Point,
    projected_body: &[Point],
    my_length: i32,
    wraps: bool,
) -> TerritoryAnalysis {
    let mut blocked: HashSet<Point> = projected_body.iter().copied().collect();
    for snake in board.snakes.iter().filter(|snake| snake.id != you.id) {
        blocked.extend(snake.body.iter().map(Point::from));
    }

    let no_danger = HashSet::new();
    let (_, our_distances) = reachable_space(board, my_target, &blocked, &no_danger, wraps);
    let enemy_distances: Vec<(i32, HashMap<Point, i32>)> = board
        .snakes
        .iter()
        .filter(|snake| snake.id != you.id)
        .map(|snake| {
            let (_, distances) =
                reachable_space(board, Point::from(&snake.head), &blocked, &no_danger, wraps);
            (snake_length(snake), distances)
        })
        .collect();

    let controlled = our_distances
        .iter()
        .filter(|(point, our_distance)| {
            let closest_enemy = enemy_distances
                .iter()
                .filter_map(|(length, distances)| {
                    distances.get(point).map(|distance| (*length, *distance))
                })
                .min_by_key(|(_, distance)| *distance);

            match closest_enemy {
                None => true,
                Some((_length, distance)) if **our_distance < distance => true,
                Some((length, distance)) if **our_distance == distance => my_length > length,
                Some(_) => false,
            }
        })
        .count() as i64;

    TerritoryAnalysis {
        controlled,
        enemy_distances,
    }
}

/// Return true only for a head-to-head kill the shorter snake cannot decline.
///
/// This is intentionally stricter than "we are longer": attacking a shorter
/// head with open side exits merely invites it to turn away while we sacrifice
/// position. A forced kill is useful because it removes a rival and frees its
/// territory without adding uncertainty to our survival plan.
fn forced_kill_available(context: &AttackContext<'_>, my_target: Point, my_length: i32) -> bool {
    context
        .board
        .snakes
        .iter()
        .filter(|snake| snake.id != context.you.id && snake_length(snake) < my_length)
        .any(|snake| {
            can_reach_in_one_turn(snake, my_target, context.board, context.wraps)
                && opponent_escape_count(context, snake, my_target) == 0
        })
}

fn opponent_escape_count(
    context: &AttackContext<'_>,
    opponent: &Battlesnake,
    my_target: Point,
) -> usize {
    let head = Point::from(&opponent.head);
    let neck = opponent.body.get(1).map(Point::from);

    DIRECTIONS
        .iter()
        .copied()
        .filter_map(|direction| direction.next(head, context.board, context.wraps))
        .filter(|target| *target != my_target && neck != Some(*target))
        .filter(|target| {
            let grows = is_constrictor(context.game) || context.food.contains(target);
            let own_body_end = if grows {
                opponent.body.len()
            } else {
                opponent.body.len().saturating_sub(1)
            };
            if opponent.body[..own_body_end]
                .iter()
                .any(|segment| Point::from(segment) == *target)
            {
                return false;
            }

            if context.projected_body.contains(target) {
                return false;
            }

            context
                .board
                .snakes
                .iter()
                .filter(|snake| snake.id != opponent.id)
                .all(|snake| {
                    if snake.id == context.you.id {
                        true
                    } else {
                        !snake
                            .body
                            .iter()
                            .any(|segment| Point::from(segment) == *target)
                    }
                })
        })
        .count()
}

/// Explore our own legal continuation after this candidate move.
///
/// The one-turn flood fill is deliberately optimistic about a far-away tail:
/// it can call a distant region reachable even though the head must traverse a
/// narrow corridor first. This bounded beam search models each tail movement
/// and food growth, so a route that closes before the tail opens gets a low
/// score. Rival bodies stay blocked and stronger rival head territory stays
/// forbidden; that pessimism is intentional for a survival-first snake.
fn future_survival_score(
    context: &SurvivalContext<'_>,
    projected_body: &[Point],
    health_after: i32,
    first_target: Point,
    ate_first_food: bool,
) -> i64 {
    let constrictor = is_constrictor(context.game);
    let fixed_opponents: HashSet<Point> = context
        .board
        .snakes
        .iter()
        .filter(|snake| snake.id != context.you.id)
        .flat_map(|snake| snake.body.iter().map(Point::from))
        .collect();
    let initial_eaten = if ate_first_food {
        food_mask(first_target, &context.board.food)
    } else {
        0
    };
    let mut frontier = vec![SimState {
        body: projected_body.to_vec(),
        health: health_after,
        eaten_food: initial_eaten,
    }];
    let mut deepest = 0_i64;
    let mut breadth = 0_i64;

    for future_step in 0..SURVIVAL_HORIZON {
        let mut next_frontier = Vec::new();
        for state in &frontier {
            let head = state.body[0];
            for direction in DIRECTIONS.iter().copied() {
                let Some(target) = direction.next(head, context.board, context.wraps) else {
                    continue;
                };
                if fixed_opponents.contains(&target) || context.dangerous.contains(&target) {
                    continue;
                }

                let food_bit = food_mask(target, &context.board.food);
                let eating = food_bit != 0 && state.eaten_food & food_bit == 0;
                let grows = constrictor || eating;
                let next_length = state.body.len() as i32 + i32::from(grows);
                let enemy_can_arrive =
                    context
                        .enemy_arrivals
                        .iter()
                        .any(|(enemy_length, distances)| {
                            *enemy_length + i32::from(food_bit != 0) >= next_length
                                && distances.get(&target).is_some_and(|enemy_turns| {
                                    *enemy_turns <= future_step as i32 + 2
                                })
                        });
                if enemy_can_arrive {
                    continue;
                }
                let own_body_end = if grows {
                    state.body.len()
                } else {
                    state.body.len().saturating_sub(1)
                };
                if state.body[..own_body_end].contains(&target) {
                    continue;
                }

                let hazard_cost = if grows {
                    0
                } else {
                    context.hazard_stacks.get(&target).copied().unwrap_or(0)
                        * hazard_damage(context.game)
                };
                let next_health = if constrictor || eating {
                    100
                } else {
                    state.health - 1 - hazard_cost
                };
                if next_health <= 0 {
                    continue;
                }

                let mut body = Vec::with_capacity(state.body.len() + 1);
                body.push(target);
                body.extend(state.body.iter().copied());
                if !grows {
                    body.pop();
                }
                next_frontier.push(SimState {
                    body,
                    health: next_health,
                    eaten_food: state.eaten_food | food_bit,
                });
            }
        }

        if next_frontier.is_empty() {
            break;
        }
        deepest += 1;
        breadth += next_frontier.len().min(SURVIVAL_BEAM_WIDTH) as i64;
        next_frontier.sort_unstable_by(|a, b| {
            simulation_quality(
                b,
                context.board,
                &fixed_opponents,
                context.dangerous,
                context.wraps,
            )
            .cmp(&simulation_quality(
                a,
                context.board,
                &fixed_opponents,
                context.dangerous,
                context.wraps,
            ))
        });
        next_frontier.truncate(SURVIVAL_BEAM_WIDTH);
        frontier = next_frontier;
    }

    // A deep forced tail chase can work; breadth is the secondary signal that
    // favours positions with several exits instead of a single brittle line.
    deepest * 700 + breadth * 4
}

fn food_mask(point: Point, food: &[Coord]) -> u128 {
    food.iter()
        .position(|food| Point::from(food) == point)
        .filter(|index| *index < 128)
        .map(|index| 1_u128 << index)
        .unwrap_or(0)
}

fn simulation_quality(
    state: &SimState,
    board: &Board,
    fixed_opponents: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    wraps: bool,
) -> i64 {
    let head = state.body[0];
    let own_body_end = state.body.len().saturating_sub(1);
    let exits = DIRECTIONS
        .iter()
        .filter_map(|direction| direction.next(head, board, wraps))
        .filter(|point| {
            !fixed_opponents.contains(point)
                && !dangerous.contains(point)
                && !state.body[..own_body_end].contains(point)
        })
        .count() as i64;
    exits * 100 + centre_score(head, board) + state.health as i64
}

fn reachable_space(
    board: &Board,
    start: Point,
    blocked: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    wraps: bool,
) -> (usize, HashMap<Point, i32>) {
    let mut distances = HashMap::new();
    let mut queue = VecDeque::new();
    distances.insert(start, 0);
    queue.push_back(start);

    while let Some(point) = queue.pop_front() {
        let distance = distances[&point];
        for direction in DIRECTIONS.iter().copied() {
            let Some(next) = direction.next(point, board, wraps) else {
                continue;
            };
            if blocked.contains(&next) || dangerous.contains(&next) || distances.contains_key(&next)
            {
                continue;
            }
            distances.insert(next, distance + 1);
            queue.push_back(next);
        }
    }

    (distances.len(), distances)
}


/// Detect corridor/trap geometry close to the head.
///
/// Total reachable space alone can be misleading: a large area may only be
/// reachable through a one-cell corridor. This metric therefore looks at
/// immediate exits, available room relative to body length, and narrow cells
/// in the first few BFS layers.
fn trap_risk_score(
    board: &Board,
    start: Point,
    blocked: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    wraps: bool,
    body_len: usize,
) -> i64 {
    let (space, distances) = reachable_space(board, start, blocked, dangerous, wraps);

    let immediate_exits = DIRECTIONS
        .iter()
        .filter_map(|direction| direction.next(start, board, wraps))
        .filter(|point| !blocked.contains(point) && !dangerous.contains(point))
        .count();

    let mut risk = match immediate_exits {
        0 => 5_000,
        1 => 2_000,
        2 => 250,
        _ => 0,
    };

    let minimum_space = body_len.saturating_add(4);
    if space < minimum_space {
        risk += (minimum_space - space) as i64 * 350;
    }

    let mut narrow = 0_i64;

    for (point, distance) in &distances {
        if *distance > TRAP_LOOKAHEAD {
            continue;
        }

        let exits = DIRECTIONS
            .iter()
            .filter_map(|direction| direction.next(*point, board, wraps))
            .filter(|next| !blocked.contains(next) && !dangerous.contains(next))
            .count();

        if exits <= 1 {
            narrow += 1;
        }
    }

    risk + narrow.min(20) * 120
}

/// Estimate how quickly enemy heads can reach the area we are moving into.
fn enemy_pressure_score(
    board: &Board,
    you: &Battlesnake,
    target: Point,
    wraps: bool,
) -> i64 {
    let mut pressure = 0_i64;

    for enemy in &board.snakes {
        if enemy.id == you.id {
            continue;
        }

        let head = Point::from(&enemy.head);
        let neck = enemy.body.get(1).map(Point::from);

        let mut blocked = HashSet::new();

        // Block the enemy's own body except its head, plus every other snake.
        // This makes the BFS represent actual paths the enemy head can take.
        blocked.extend(enemy.body.iter().skip(1).map(Point::from));

        for snake in &board.snakes {
            if snake.id != enemy.id {
                blocked.extend(snake.body.iter().map(Point::from));
            }
        }

        if let Some(neck) = neck {
            blocked.remove(&neck);
        }

        let (_, distances) = reachable_space(board, head, &blocked, &HashSet::new(), wraps);

        let Some(&distance) = distances.get(&target) else {
            continue;
        };

        if distance <= PRESSURE_DISTANCE {
            let proximity = (PRESSURE_DISTANCE - distance + 1) as i64;
            let size_factor = if snake_length(enemy) >= snake_length(you) {
                3
            } else {
                1
            };

            pressure += proximity * proximity * size_factor;
        }
    }

    pressure
}

#[derive(Debug, Clone, Copy)]
struct AdversarialAnalysis {
    worst_case_space: usize,
    worst_case_exits: i64,
    escape_routes: i64,
    future_worst_exits: i64,
    future_escape_routes: i64,
    enemy_cutoff_risk: i64,
    enemy_push_risk: i64,
    edge_exposure_risk: i64,
    commitment_risk: i64,
}

#[derive(Debug, Clone, Copy)]
struct EscapeFollowup {
    head: Point,
    exits: i64,
    escape_routes: i64,
    space: usize,
    boundary_distance: i64,
}

/// Look one move ahead for the opponents and ask a deliberately adversarial
/// question: "after I move here, what can the opponent do that most reduces
/// my escape routes?"
///
/// This is not a full minimax. Each enemy is examined independently with its
/// legal one-step responses. That keeps the cost small on tournament boards
/// while still detecting the exact failure mode we saw in replays: plenty of
/// global space, but the enemy can take away the only useful exit and push us
/// into the boundary.
fn adversarial_escape_analysis(
    game: &Game,
    board: &Board,
    you: &Battlesnake,
    target: Point,
    projected_body: &[Point],
    dangerous: &HashSet<Point>,
    hazard_stacks: &HashMap<Point, i32>,
    wraps: bool,
    base_exits: i64,
    base_space: usize,
    health_after: i32,
) -> AdversarialAnalysis {
    let mut worst_space = base_space;
    let mut worst_exits = base_exits;
    let mut worst_escape_routes = base_exits;
    let mut future_worst_exits = base_exits;
    let mut future_escape_routes = base_exits;
    let mut cutoff_risk = 0_i64;
    let mut push_risk = 0_i64;
    let mut commitment_risk = 0_i64;

    for enemy in board.snakes.iter().filter(|snake| snake.id != you.id) {
        let enemy_moves = legal_enemy_projection(
            game,
            board,
            you,
            enemy,
            target,
            projected_body,
            hazard_stacks,
            wraps,
        );

        if enemy_moves.is_empty() {
            continue;
        }

        let mut enemy_worst_space = usize::MAX;
        let mut enemy_worst_exits = i64::MAX;
        let mut enemy_worst_escape_routes = i64::MAX;
        let mut enemy_worst_future_exits = i64::MAX;
        let mut enemy_worst_future_escape = i64::MAX;
        let mut enemy_best_push_risk = 0_i64;
        let mut enemy_max_commitment = 0_i64;

        for projected_enemy in enemy_moves {
            let mut blocked = HashSet::new();
            blocked.extend(projected_body.iter().copied());

            for snake in board.snakes.iter().filter(|snake| snake.id != you.id) {
                if snake.id == enemy.id {
                    continue;
                }
                blocked.extend(snake.body.iter().map(Point::from));
            }
            blocked.extend(projected_enemy.iter().copied());

            // On our next move our tail normally vacates. Keep that optimistic
            // tail access unless we are in Constrictor, where the tail is fixed.
            if !is_constrictor(game) {
                if let Some(tail) = projected_body.last().copied() {
                    blocked.remove(&tail);
                }
            }

            let exits = count_escape_exits(board, target, &blocked, dangerous, wraps) as i64;
            let escape_routes = escape_routes_away_from_enemies(
                board,
                target,
                &blocked,
                dangerous,
                projected_enemy[0],
                wraps,
            ) as i64;
            let (space, _) = reachable_space(board, target, &blocked, dangerous, wraps);
            let push = enemy_push_risk(
                board,
                target,
                &blocked,
                dangerous,
                projected_enemy[0],
                exits,
                escape_routes,
                wraps,
            );

            // V2.3: after the enemy responds, explicitly inspect our best
            // defensive continuation. This prevents a false sense of safety
            // where the current square still has 2-3 exits but every useful
            // second move leads into a one-exit lane.
            let followup = best_escape_after_enemy(
                game,
                board,
                you,
                target,
                projected_body,
                health_after,
                &projected_enemy,
                enemy.id.as_str(),
                &blocked,
                dangerous,
                hazard_stacks,
                wraps,
            );

            let temporal_push = temporal_push_risk(
                board,
                target,
                projected_enemy[0],
                followup,
                push,
                wraps,
            );

            let commitment = commitment_risk_score(
                target,
                board,
                exits,
                escape_routes,
                followup,
                push.max(temporal_push),
                wraps,
            );

            enemy_worst_space = enemy_worst_space.min(space);
            enemy_worst_exits = enemy_worst_exits.min(exits);
            enemy_worst_escape_routes = enemy_worst_escape_routes.min(escape_routes);
            enemy_worst_future_exits = enemy_worst_future_exits.min(followup.exits);
            enemy_worst_future_escape = enemy_worst_future_escape.min(followup.escape_routes);
            enemy_best_push_risk = enemy_best_push_risk.max(push.max(temporal_push));
            enemy_max_commitment = enemy_max_commitment.max(commitment);
        }

        if enemy_worst_space != usize::MAX {
            worst_space = worst_space.min(enemy_worst_space);
            worst_exits = worst_exits.min(enemy_worst_exits);
            worst_escape_routes = worst_escape_routes.min(enemy_worst_escape_routes);
            future_worst_exits = future_worst_exits.min(enemy_worst_future_exits);
            future_escape_routes = future_escape_routes.min(enemy_worst_future_escape);

            let exit_loss = (base_exits - enemy_worst_exits).max(0);
            let forced_one_exit = if enemy_worst_exits <= 1 { 1 } else { 0 };
            let future_forced_one_exit = if enemy_worst_future_exits <= 1 { 1 } else { 0 };
            let future_escape_loss = (MIN_STRATEGIC_ESCAPE_ROUTES - enemy_worst_future_escape).max(0);
            cutoff_risk = cutoff_risk.max(
                exit_loss * 550
                    + forced_one_exit * 2_500
                    + future_forced_one_exit * 2_500
                    + future_escape_loss * 650
                    + ((base_space as i64 - enemy_worst_space as i64).max(0) * 6),
            );
            push_risk = push_risk.max(enemy_best_push_risk);
            commitment_risk = commitment_risk.max(enemy_max_commitment);
        }
    }

    let edge_exposure_risk = boundary_exposure_risk(
        board,
        target,
        worst_exits.min(future_worst_exits),
        push_risk,
        wraps,
    );

    AdversarialAnalysis {
        worst_case_space: worst_space,
        worst_case_exits: worst_exits,
        escape_routes: worst_escape_routes,
        future_worst_exits,
        future_escape_routes,
        enemy_cutoff_risk: cutoff_risk,
        enemy_push_risk: push_risk,
        edge_exposure_risk,
        commitment_risk,
    }
}

fn best_escape_after_enemy(
    game: &Game,
    board: &Board,
    you: &Battlesnake,
    target: Point,
    projected_body: &[Point],
    health_after_first: i32,
    projected_enemy: &[Point],
    primary_enemy_id: &str,
    blocked_after_enemy: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    hazard_stacks: &HashMap<Point, i32>,
    wraps: bool,
) -> EscapeFollowup {
    let food = point_set(&board.food);
    let my_head_len = projected_body.len() as i32;
    let enemy_length = projected_enemy.len() as i32;
    let mut best: Option<EscapeFollowup> = None;

    for direction in DIRECTIONS.iter().copied() {
        let Some(next_head) = direction.next(target, board, wraps) else {
            continue;
        };
        if blocked_after_enemy.contains(&next_head) && Some(next_head) != projected_body.last().copied() {
            continue;
        }

        let grows = is_constrictor(game) || food.contains(&next_head);
        let own_body_end = if grows {
            projected_body.len()
        } else {
            projected_body.len().saturating_sub(1)
        };
        if projected_body[..own_body_end].contains(&next_head) {
            continue;
        }

        if next_head == projected_enemy[0] && enemy_length >= my_head_len + i32::from(grows) {
            continue;
        }

        let hazard_cost = if grows {
            0
        } else {
            hazard_stacks.get(&next_head).copied().unwrap_or(0) * hazard_damage(game)
        };
        let next_health = if grows {
            100
        } else {
            health_after_first - 1 - hazard_cost
        };
        // Do not call a move an "escape" if it actually dies on the way.
        if next_health <= 0 {
            continue;
        }

        let mut body = Vec::with_capacity(projected_body.len() + 1);
        body.push(next_head);
        body.extend(projected_body.iter().copied());
        if !grows {
            body.pop();
        }

        let mut blocked = HashSet::new();
        for snake in board.snakes.iter().filter(|snake| snake.id != you.id) {
            if snake.id == primary_enemy_id {
                continue;
            }
            blocked.extend(snake.body.iter().map(Point::from));
        }
        blocked.extend(projected_enemy.iter().copied());
        blocked.extend(body.iter().copied());
        if !is_constrictor(game) {
            if let Some(tail) = body.last().copied() {
                blocked.remove(&tail);
            }
        }

        let exits = count_escape_exits(board, next_head, &blocked, dangerous, wraps) as i64;
        let escape_routes = escape_routes_away_from_enemies(
            board,
            next_head,
            &blocked,
            dangerous,
            projected_enemy[0],
            wraps,
        ) as i64;
        let (space, _) = reachable_space(board, next_head, &blocked, dangerous, wraps);
        let boundary = boundary_distance(next_head, board);

        let current = EscapeFollowup {
            head: next_head,
            exits,
            escape_routes,
            space,
            boundary_distance: boundary,
        };

        let is_better = best.is_none_or(|previous| {
            (current.exits, current.escape_routes, current.boundary_distance, current.space.min(200))
                > (previous.exits, previous.escape_routes, previous.boundary_distance, previous.space.min(200))
        });

        if is_better {
            best = Some(current);
        }
    }

    best.unwrap_or(EscapeFollowup {
        head: target,
        exits: 0,
        escape_routes: 0,
        space: 0,
        boundary_distance: boundary_distance(target, board),
    })
}

fn temporal_push_risk(
    board: &Board,
    target: Point,
    enemy_head: Point,
    followup: EscapeFollowup,
    current_push: i64,
    wraps: bool,
) -> i64 {
    let current_distance = topology_distance(target, enemy_head, board, wraps);
    let follow_distance = topology_distance(followup.head, enemy_head, board, wraps);
    let current_edge = boundary_distance(target, board);
    let follow_edge = boundary_distance(followup.head, board);
    let mut risk = current_push;

    if current_push > 0 && follow_distance <= current_distance && follow_edge <= current_edge {
        risk += 900;
    }
    if current_push > 0 && followup.exits <= 1 {
        risk += 1_200;
    }
    if current_push > 0 && followup.escape_routes == 0 {
        risk += 900;
    }
    risk
}

fn commitment_risk_score(
    target: Point,
    board: &Board,
    exits: i64,
    escape_routes: i64,
    followup: EscapeFollowup,
    push: i64,
    wraps: bool,
) -> i64 {
    let edge = boundary_distance(target, board);
    let mut risk = 0_i64;

    if followup.exits <= 1 {
        risk += 2_000;
    } else if followup.exits == 2 {
        risk += 500;
    }

    if followup.escape_routes == 0 {
        risk += 1_400;
    } else if followup.escape_routes == 1 {
        risk += 500;
    }

    if escape_routes <= 0 {
        risk += 800;
    } else if escape_routes == 1 {
        risk += 250;
    }

    if push >= ENEMY_PUSH_DANGER {
        risk += 1_000;
    }
    if push >= ENEMY_PUSH_DANGER && followup.exits <= 2 {
        risk += 1_200;
    }
    if push >= ENEMY_PUSH_DANGER && followup.escape_routes <= 1 {
        risk += 900;
    }

    if !wraps && edge <= 1 {
        if exits <= 2 {
            risk += 700;
        }
        if followup.boundary_distance <= edge {
            risk += 650;
        }
    }

    risk
}

fn is_cerco_risk(candidate: &Candidate) -> bool {
    is_cerco_risk_values(&AdversarialAnalysis {
        worst_case_space: candidate.worst_case_space,
        worst_case_exits: candidate.worst_case_exits,
        escape_routes: candidate.escape_routes,
        future_worst_exits: candidate.future_worst_exits,
        future_escape_routes: candidate.future_escape_routes,
        enemy_cutoff_risk: candidate.enemy_cutoff_risk,
        enemy_push_risk: candidate.enemy_push_risk,
        edge_exposure_risk: candidate.edge_exposure_risk,
        commitment_risk: candidate.commitment_risk,
    })
}

fn is_cerco_risk_values(analysis: &AdversarialAnalysis) -> bool {
    if analysis.future_worst_exits <= 1
        && (analysis.enemy_push_risk >= ENEMY_PUSH_DANGER
            || analysis.enemy_cutoff_risk >= ENEMY_CUTOFF_DANGER
            || analysis.edge_exposure_risk >= EDGE_DANGER)
    {
        return true;
    }

    if analysis.future_escape_routes <= 1 && analysis.enemy_push_risk >= ENEMY_PUSH_DANGER {
        return true;
    }

    if analysis.enemy_push_risk >= ENEMY_PUSH_DANGER
        && analysis.future_worst_exits <= 2
        && analysis.future_escape_routes <= 2
    {
        return true;
    }

    if analysis.edge_exposure_risk >= EDGE_DANGER
        && analysis.worst_case_exits <= 2
        && analysis.future_worst_exits <= 2
    {
        return true;
    }

    analysis.commitment_risk >= 3_500
}

fn legal_enemy_projection(
    game: &Game,
    board: &Board,
    you: &Battlesnake,
    enemy: &Battlesnake,
    our_target: Point,
    our_projected_body: &[Point],
    hazard_stacks: &HashMap<Point, i32>,
    wraps: bool,
) -> Vec<Vec<Point>> {
    let enemy_head = Point::from(&enemy.head);
    let enemy_neck = enemy.body.get(1).map(Point::from);
    let food = point_set(&board.food);
    let my_length = snake_length(you);
    let mut projections = Vec::new();

    for direction in DIRECTIONS.iter().copied() {
        let Some(target) = direction.next(enemy_head, board, wraps) else {
            continue;
        };
        if enemy_neck == Some(target) {
            continue;
        }

        // Do not treat a move directly onto our projected head as a useful
        // blocking response unless the enemy can actually survive that H2H.
        if target == our_target {
            let enemy_length = snake_length(enemy);
            if enemy_length <= my_length {
                continue;
            }
        }

        let grows = is_constrictor(game) || food.contains(&target);
        let body_end = if grows {
            enemy.body.len()
        } else {
            enemy.body.len().saturating_sub(1)
        };
        if enemy.body[..body_end]
            .iter()
            .any(|segment| Point::from(segment) == target)
        {
            continue;
        }

        if board.snakes.iter().any(|snake| {
            snake.id != enemy.id
                && snake
                    .body
                    .iter()
                    .any(|segment| Point::from(segment) == target)
                && snake.id != you.id
        }) {
            continue;
        }

        if our_projected_body.contains(&target) && target != our_target {
            continue;
        }

        let hazard_cost = if grows {
            0
        } else {
            hazard_stacks.get(&target).copied().unwrap_or(0) * hazard_damage(game)
        };
        let health_after = if grows {
            100
        } else {
            enemy.health - 1 - hazard_cost
        };
        if health_after <= 0 {
            continue;
        }

        let mut projected = Vec::with_capacity(enemy.body.len() + 1);
        projected.push(target);
        projected.extend(enemy.body.iter().map(Point::from));
        if !grows {
            projected.pop();
        }
        projections.push(projected);
    }

    projections
}

fn count_escape_exits(
    board: &Board,
    start: Point,
    blocked: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    wraps: bool,
) -> usize {
    DIRECTIONS
        .iter()
        .filter_map(|direction| direction.next(start, board, wraps))
        .filter(|point| !blocked.contains(point) && !dangerous.contains(point))
        .count()
}

fn escape_routes_away_from_enemies(
    board: &Board,
    target: Point,
    blocked: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    primary_enemy_head: Point,
    wraps: bool,
) -> usize {
    let target_distance = topology_distance(target, primary_enemy_head, board, wraps);

    DIRECTIONS
        .iter()
        .filter_map(|direction| direction.next(target, board, wraps))
        .filter(|point| !blocked.contains(point) && !dangerous.contains(point))
        .filter(|point| topology_distance(*point, primary_enemy_head, board, wraps) > target_distance)
        .count()
}

fn enemy_push_risk(
    board: &Board,
    target: Point,
    blocked: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    enemy_head: Point,
    exits: i64,
    escape_routes: i64,
    wraps: bool,
) -> i64 {
    let distance = topology_distance(target, enemy_head, board, wraps);
    let mut risk = 0_i64;

    if distance <= PUSH_DISTANCE_THRESHOLD {
        let proximity = (PUSH_DISTANCE_THRESHOLD - distance + 1).max(1);
        if escape_routes == 0 {
            risk += proximity * 180;
        } else if escape_routes == 1 {
            risk += proximity * 55;
        }
    }

    if exits <= 1 {
        risk += 1_200;
    } else if exits == 2 && escape_routes == 0 {
        risk += 550;
    }

    let interior_escape = DIRECTIONS
        .iter()
        .filter_map(|direction| direction.next(target, board, wraps))
        .filter(|point| !blocked.contains(point) && !dangerous.contains(point))
        .filter(|point| boundary_distance(*point, board) > boundary_distance(target, board))
        .count() as i64;

    if boundary_distance(target, board) <= 1 && exits <= 2 && interior_escape == 0 {
        risk += 850;
    }

    risk
}

fn boundary_exposure_risk(
    board: &Board,
    target: Point,
    worst_exits: i64,
    push_risk: i64,
    wraps: bool,
) -> i64 {
    if wraps {
        return 0;
    }

    let edge = boundary_distance(target, board);
    let mut risk = 0_i64;

    if edge == 0 {
        if worst_exits <= 1 {
            risk += 2_000;
        } else if worst_exits == 2 {
            risk += 700;
        }
    } else if edge == 1 {
        if worst_exits <= 1 {
            risk += 1_200;
        } else if worst_exits == 2 {
            risk += 300;
        }
    }

    if push_risk > 0 && edge <= 2 {
        risk += (3 - edge) * 180;
    }

    risk
}

fn boundary_distance(point: Point, board: &Board) -> i64 {
    let left = point.x.max(0);
    let right = (board.width - 1 - point.x).max(0);
    let bottom = point.y.max(0);
    let top = (board.height - 1 - point.y).max(0);
    left.min(right).min(bottom).min(top) as i64
}

fn topology_distance(a: Point, b: Point, board: &Board, wraps: bool) -> i64 {
    let dx = (a.x - b.x).abs() as i64;
    let dy = (a.y - b.y).abs() as i64;

    if wraps {
        let width = board.width.max(1) as i64;
        let height = board.height.max(1) as i64;
        let wrapped_dx = dx.rem_euclid(width);
        let wrapped_dy = dy.rem_euclid(height);
        wrapped_dx.min(width - wrapped_dx) + wrapped_dy.min(height - wrapped_dy)
    } else {
        dx + dy
    }
}

/// Score the first few BFS layers so a wide room beats a long narrow tunnel.
fn future_space_score(
    board: &Board,
    start: Point,
    blocked: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    wraps: bool,
    body_len: usize,
) -> i64 {
    let (_, distances) = reachable_space(board, start, blocked, dangerous, wraps);

    let mut score = 0_i64;
    let mut previous_frontier = 1_i64;

    for depth in 1..=6_i32 {
        let frontier = distances
            .values()
            .filter(|distance| **distance == depth)
            .count() as i64;

        if frontier == 0 {
            if depth <= 3 && body_len > depth as usize + 2 {
                score -= (7 - depth) as i64 * 120;
            }
            break;
        }

        score += frontier * depth as i64;

        if depth > 1 && frontier * 2 < previous_frontier {
            score -= (previous_frontier - frontier) * 80;
        }

        previous_frontier = frontier;
    }

    score
}

fn nearest_food(
    food: &HashSet<Point>,
    current_target: Point,
    my_length: i32,
    distances: &HashMap<Point, i32>,
    enemy_distances: &[(i32, HashMap<Point, i32>)],
) -> (Option<i32>, bool) {
    let mut closest: Option<(i32, bool)> = None;
    for point in food
        .iter()
        .copied()
        .filter(|point| *point != current_target)
    {
        let Some(&distance) = distances.get(&point) else {
            continue;
        };
        // Add the move already selected when comparing race times from the
        // current board state. Both sides use real board paths rather than a
        // Manhattan estimate, so walls and coiled bodies matter in the race.
        let my_turns = distance + 1;
        let contested = enemy_distances.iter().any(|(enemy_length, distances)| {
            *enemy_length >= my_length
                && distances
                    .get(&point)
                    .is_some_and(|enemy_turns| *enemy_turns <= my_turns)
        });
        if closest.is_none_or(|(best, _)| distance < best) {
            closest = Some((distance, contested));
        }
    }
    closest.map_or((None, false), |(distance, contested)| {
        (Some(distance), contested)
    })
}

fn centre_score(point: Point, board: &Board) -> i64 {
    // Prefer the centre only as a tie-breaker; usable space dominates this.
    let horizontal = (2 * point.x - (board.width - 1)).abs();
    let vertical = (2 * point.y - (board.height - 1)).abs();
    -(horizontal + vertical) as i64 * 8
}

fn ruleset_name(game: &Game) -> &str {
    game.ruleset
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
}

fn is_constrictor(game: &Game) -> bool {
    ruleset_name(game).eq_ignore_ascii_case("constrictor")
}

fn is_wrapped(game: &Game) -> bool {
    game.map
        .as_deref()
        .map(|map| map.eq_ignore_ascii_case("wrapped"))
        .unwrap_or(false)
        || ruleset_name(game).eq_ignore_ascii_case("wrapped")
}

fn hazard_damage(game: &Game) -> i32 {
    game.ruleset
        .get("settings")
        .and_then(|settings| settings.get("hazardDamagePerTurn"))
        .and_then(Value::as_i64)
        .and_then(|damage| i32::try_from(damage).ok())
        .unwrap_or(14)
}

// move is called on every turn and returns the next move.
pub fn get_move(game: &Game, turn: &i32, board: &Board, you: &Battlesnake) -> Value {
    let started = Instant::now();
    let chosen = select_direction(game, board, you);
    let elapsed_ms = started.elapsed().as_millis();
    let warning_threshold_ms = u128::from(game.timeout) * 3 / 4;
    let target = chosen
        .next(Point::from(&you.head), board, is_wrapped(game))
        .map(|point| format!("({}, {})", point.x, point.y))
        .unwrap_or_else(|| "outside-board".to_owned());

    if elapsed_ms >= warning_threshold_ms {
        warn!(
            "MOVE_SLOW id={} turn={} head=({}, {}) direction={} target={} compute_ms={} timeout_ms={}",
            game.id,
            turn,
            you.head.x,
            you.head.y,
            chosen.name(),
            target,
            elapsed_ms,
            game.timeout
        );
    } else {
        info!(
            "MOVE id={} turn={} head=({}, {}) direction={} target={} compute_ms={} timeout_ms={}",
            game.id,
            turn,
            you.head.x,
            you.head.y,
            chosen.name(),
            target,
            elapsed_ms,
            game.timeout
        );
    }
    json!({ "move": chosen.name() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn coord(x: i32, y: i32) -> Coord {
        Coord { x, y }
    }

    fn snake(id: &str, health: i32, body: &[(i32, i32)]) -> Battlesnake {
        let body: Vec<Coord> = body.iter().map(|&(x, y)| coord(x, y)).collect();
        Battlesnake {
            id: id.to_owned(),
            name: id.to_owned(),
            health,
            head: body[0],
            length: body.len() as i32,
            body,
            latency: "0".to_owned(),
            shout: None,
        }
    }

    fn game(ruleset_name: &str, map: Option<&str>) -> Game {
        let mut ruleset = HashMap::new();
        ruleset.insert("name".to_owned(), Value::String(ruleset_name.to_owned()));
        ruleset.insert("settings".to_owned(), json!({ "hazardDamagePerTurn": 14 }));
        Game {
            id: "test".to_owned(),
            ruleset,
            map: map.map(str::to_owned),
            timeout: 500,
        }
    }

    fn board(
        width: i32,
        height: i32,
        food: &[(i32, i32)],
        hazards: &[(i32, i32)],
        snakes: Vec<Battlesnake>,
    ) -> Board {
        Board {
            width,
            height,
            food: food.iter().map(|&(x, y)| coord(x, y)).collect(),
            hazards: hazards.iter().map(|&(x, y)| coord(x, y)).collect(),
            snakes,
        }
    }

    #[test]
    fn never_reverses_or_leaves_the_board() {
        let me = snake("me", 100, &[(0, 1), (1, 1), (2, 1)]);
        let board = board(
            3,
            3,
            &[],
            &[],
            vec![snake("me", 100, &[(0, 1), (1, 1), (2, 1)])],
        );

        let direction = select_direction(&game("standard", None), &board, &me);

        assert!(matches!(direction, Direction::Up | Direction::Down));
    }

    #[test]
    fn avoids_equal_length_head_to_head() {
        let me = snake("me", 100, &[(1, 3), (1, 2), (1, 1)]);
        let enemy = snake("enemy", 100, &[(3, 3), (3, 2), (3, 1)]);
        let board = board(
            7,
            7,
            &[],
            &[],
            vec![snake("me", 100, &[(1, 3), (1, 2), (1, 1)]), enemy],
        );

        assert_ne!(
            select_direction(&game("standard", None), &board, &me),
            Direction::Right
        );
    }

    #[test]
    fn avoids_a_single_exit_trap_when_open_space_exists() {
        let me = snake("me", 90, &[(2, 2), (2, 1), (1, 1), (1, 2)]);
        let board = board(
            7,
            7,
            &[],
            &[],
            vec![snake("me", 90, &[(2, 2), (2, 1), (1, 1), (1, 2)])],
        );

        assert_ne!(
            select_direction(&game("standard", None), &board, &me),
            Direction::Down
        );
    }

    #[test]
    fn rescues_low_health_with_reachable_food() {
        let me = snake("me", 20, &[(2, 2), (2, 1)]);
        let board = board(
            7,
            7,
            &[(3, 2)],
            &[],
            vec![snake("me", 20, &[(2, 2), (2, 1)])],
        );

        assert_eq!(
            select_direction(&game("standard", None), &board, &me),
            Direction::Right
        );
    }

    #[test]
    fn eats_adjacent_food_when_starving() {
        let me = snake("me", 2, &[(2, 2), (2, 1), (2, 0)]);
        let board = board(
            7,
            7,
            &[(3, 2)],
            &[],
            vec![snake("me", 2, &[(2, 2), (2, 1), (2, 0)])],
        );

        assert_eq!(
            select_direction(&game("standard", None), &board, &me),
            Direction::Right
        );
    }

    #[test]
    fn permits_following_our_own_vacating_tail() {
        let me = snake("me", 10, &[(1, 1), (1, 2), (0, 2), (0, 1)]);
        let enemy = snake("enemy", 100, &[(2, 1), (2, 0)]);
        let board = board(
            3,
            3,
            &[],
            &[(1, 0)],
            vec![snake("me", 10, &[(1, 1), (1, 2), (0, 2), (0, 1)]), enemy],
        );

        assert_eq!(
            select_direction(&game("standard", None), &board, &me),
            Direction::Left
        );
    }

    #[test]
    fn food_in_a_hazard_saves_a_one_health_snake() {
        let me = snake("me", 1, &[(1, 1), (1, 0)]);
        let board = board(
            5,
            5,
            &[(2, 1)],
            &[(2, 1)],
            vec![snake("me", 1, &[(1, 1), (1, 0)])],
        );

        assert_eq!(
            select_direction(&game("standard", None), &board, &me),
            Direction::Right
        );
    }

    #[test]
    fn avoids_a_lethal_hazard_when_another_move_exists() {
        let me = snake("me", 5, &[(2, 2), (2, 1)]);
        let board = board(
            5,
            5,
            &[],
            &[(3, 2)],
            vec![snake("me", 5, &[(2, 2), (2, 1)])],
        );

        assert_ne!(
            select_direction(&game("standard", None), &board, &me),
            Direction::Right
        );
    }

    #[test]
    fn wraps_at_the_board_edge_when_the_map_requires_it() {
        let me = snake("me", 5, &[(0, 2), (1, 2)]);
        let board = board(
            5,
            5,
            &[],
            &[(0, 1), (0, 3)],
            vec![
                snake("me", 5, &[(0, 2), (1, 2)]),
                snake("enemy", 100, &[(1, 2), (1, 1)]),
            ],
        );

        assert_eq!(
            select_direction(&game("standard", Some("wrapped")), &board, &me),
            Direction::Left
        );
    }

    #[test]
    fn constrictor_never_assumes_its_tail_will_move() {
        let me = snake("me", 100, &[(1, 1), (1, 2), (0, 2), (0, 1)]);
        let enemy = snake("enemy", 100, &[(2, 1), (2, 0)]);
        let board = board(
            3,
            3,
            &[],
            &[],
            vec![snake("me", 100, &[(1, 1), (1, 2), (0, 2), (0, 1)]), enemy],
        );

        assert_ne!(
            select_direction(&game("constrictor", None), &board, &me),
            Direction::Left
        );
    }

    #[test]
    fn keeps_the_projected_tail_available_for_space_estimation() {
        let me = snake("me", 100, &[(2, 2), (2, 1), (2, 0)]);
        let target = Point { x: 3, y: 2 };
        let projected = projected_body(&me, target, false);
        let (mut blocked, tail_moves) = blocked_after_move(
            &board(
                5,
                5,
                &[],
                &[],
                vec![snake("me", 100, &[(2, 2), (2, 1), (2, 0)])],
            ),
            &me,
            &projected,
            false,
        );

        assert!(tail_moves);
        let projected_tail = *projected.last().expect("a snake has a tail");
        blocked.remove(&projected_tail);
        assert!(!blocked.contains(&projected_tail));
    }

    #[test]
    fn never_marks_our_own_food_route_as_an_enemy_race() {
        let me = snake("me", 100, &[(2, 2), (2, 1)]);
        let board = board(
            7,
            7,
            &[(4, 2)],
            &[],
            vec![snake("me", 100, &[(2, 2), (2, 1)])],
        );
        let food = point_set(&board.food);
        let mut distances = HashMap::new();
        distances.insert(Point { x: 4, y: 2 }, 1);

        let (distance, contested) = nearest_food(
            &food,
            Point { x: 3, y: 2 },
            snake_length(&me),
            &distances,
            &[],
        );

        assert_eq!(distance, Some(1));
        assert!(!contested);
    }

    #[test]
    fn rejects_food_when_an_equal_enemy_has_the_same_real_path_length() {
        let food = HashSet::from([Point { x: 4, y: 2 }]);
        let mut our_distances = HashMap::new();
        our_distances.insert(Point { x: 4, y: 2 }, 1);
        let mut enemy_path = HashMap::new();
        enemy_path.insert(Point { x: 4, y: 2 }, 2);

        let (distance, contested) = nearest_food(
            &food,
            Point { x: 3, y: 2 },
            4,
            &our_distances,
            &[(4, enemy_path)],
        );

        assert_eq!(distance, Some(1));
        assert!(contested);
    }

    #[test]
    fn regression_from_log_does_not_repeat_into_the_left_wall() {
        let me = snake(
            "me",
            99,
            &[
                (0, 10),
                (1, 10),
                (1, 9),
                (2, 9),
                (2, 8),
                (2, 7),
                (2, 6),
                (1, 6),
                (0, 6),
            ],
        );
        let enemy = snake(
            "enemy",
            98,
            &[
                (7, 1),
                (7, 0),
                (6, 0),
                (6, 1),
                (6, 2),
                (6, 3),
                (5, 3),
                (4, 3),
            ],
        );
        let board = board(
            11,
            11,
            &[(10, 7), (10, 0), (0, 8), (5, 0), (3, 1), (0, 7)],
            &[],
            vec![
                snake(
                    "me",
                    99,
                    &[
                        (0, 10),
                        (1, 10),
                        (1, 9),
                        (2, 9),
                        (2, 8),
                        (2, 7),
                        (2, 6),
                        (1, 6),
                        (0, 6),
                    ],
                ),
                enemy,
            ],
        );

        assert_eq!(
            select_direction(&game("standard", None), &board, &me),
            Direction::Down
        );
    }

    #[test]
    fn takes_a_forced_head_to_head_kill_when_we_are_longer() {
        let me = snake("me", 100, &[(1, 2), (1, 1), (1, 0), (0, 0)]);
        let enemy = snake("enemy", 100, &[(3, 2), (3, 1), (3, 0)]);
        let board = board(
            5,
            3,
            &[],
            &[],
            vec![
                snake("me", 100, &[(1, 2), (1, 1), (1, 0), (0, 0)]),
                enemy,
                snake("wall", 100, &[(4, 2)]),
            ],
        );
        let target = Point { x: 2, y: 2 };
        let projected = projected_body(&me, target, false);
        let game = game("standard", None);
        let food = point_set(&board.food);
        let context = AttackContext {
            game: &game,
            board: &board,
            you: &me,
            projected_body: &projected,
            food: &food,
            wraps: false,
        };

        assert!(forced_kill_available(&context, target, snake_length(&me)));
    }

    #[test]
    fn temporal_search_detects_food_that_closes_a_pocket() {
        let me = snake("me", 100, &[(2, 2), (2, 1), (2, 0)]);
        let board = board(
            6,
            6,
            &[(3, 2)],
            &[],
            vec![
                snake("me", 100, &[(2, 2), (2, 1), (2, 0)]),
                snake("right", 100, &[(4, 2)]),
                snake("up", 100, &[(3, 3)]),
                snake("down", 100, &[(3, 1)]),
            ],
        );
        let target = Point { x: 3, y: 2 };
        let projected = projected_body(&me, target, true);
        let hazards = hazard_counts(&board.hazards);
        let dangerous = larger_head_territory(&board, &me, 4, false);
        let context = SurvivalContext {
            game: &game("standard", None),
            board: &board,
            you: &me,
            hazard_stacks: &hazards,
            dangerous: &dangerous,
            enemy_arrivals: &[],
            wraps: false,
        };

        assert_eq!(
            future_survival_score(&context, &projected, 100, target, true),
            0
        );
    }

    #[test]
    fn response_is_always_a_valid_api_move() {
        let me = snake("me", 100, &[(2, 2), (2, 1)]);
        let board = board(5, 5, &[], &[], vec![snake("me", 100, &[(2, 2), (2, 1)])]);
        let response = get_move(&game("standard", None), &7, &board, &me);

        assert!(matches!(
            response["move"].as_str(),
            Some("up" | "down" | "left" | "right")
        ));
    }


    #[test]
    fn anti_corner_prefers_a_move_with_two_enemy_resistant_exits() {
        let me = snake("me", 100, &[(2, 2), (2, 1), (1, 1)]);
        let enemy = snake("enemy", 100, &[(4, 2), (4, 1), (4, 0)]);
        let board = board(
            7,
            7,
            &[],
            &[],
            vec![
                snake("me", 100, &[(2, 2), (2, 1), (1, 1)]),
                enemy,
            ],
        );

        let chosen = select_direction(&game("standard", None), &board, &me);
        assert_ne!(chosen, Direction::Right);
    }

    #[test]
    fn detects_an_enemy_that_can_remove_our_last_escape_route() {
        // After moving to (0, 2), our two meaningful exits are up and right.
        // The enemy at (1, 3) can move to either (0, 3) or (1, 2), removing one
        // of those exits. Our old tail at (1, 0) vacates, so the fixture does
        // not rely on a tail that is incorrectly treated as permanently blocked.
        let me = snake("me", 100, &[(0, 1), (0, 0), (1, 0)]);
        let _enemy = snake("enemy", 100, &[(1, 3), (2, 3)]);
        let board = board(
            5,
            5,
            &[],
            &[],
            vec![
                snake("me", 100, &[(0, 1), (0, 0), (1, 0)]),
                snake("enemy", 100, &[(1, 3), (2, 3)]),
            ],
        );
        let target = Point { x: 0, y: 2 };
        let projected = projected_body(&me, target, false);
        let dangerous = HashSet::new();
        let hazards = hazard_counts(&board.hazards);

        let analysis = adversarial_escape_analysis(
            &game("standard", None),
            &board,
            &me,
            target,
            &projected,
            &dangerous,
            &hazards,
            false,
            2,
            20,
            90,
        );

        assert!(analysis.worst_case_exits <= 1);
        assert!(analysis.enemy_cutoff_risk > 0);
    }

    #[test]
    fn marks_a_push_toward_a_corner_as_strategic_cerco_risk() {
        let candidate = Candidate {
            direction: Direction::Down,
            score: 0,
            health_after: 90,
            space: 100,
            exits: 2,
            eating: false,
            territory: 20,
            forced_kill: false,
            future_survival: 10_000,
            trap_risk: 0,
            enemy_pressure: 0,
            adversarial_space: 100,
            future_space: 100,
            worst_case_space: 90,
            worst_case_exits: 2,
            escape_routes: 1,
            future_worst_exits: 1,
            future_escape_routes: 1,
            enemy_cutoff_risk: 2_200,
            enemy_push_risk: 2_700,
            edge_exposure_risk: 1_200,
            commitment_risk: 3_600,
            loses_head_to_head: false,
        };

        assert!(is_cerco_risk(&candidate));
    }

    #[test]
    fn does_not_mark_an_open_position_as_cerco_risk() {
        let candidate = Candidate {
            direction: Direction::Up,
            score: 0,
            health_after: 90,
            space: 100,
            exits: 3,
            eating: false,
            territory: 60,
            forced_kill: false,
            future_survival: 18_000,
            trap_risk: 0,
            enemy_pressure: 0,
            adversarial_space: 100,
            future_space: 180,
            worst_case_space: 100,
            worst_case_exits: 3,
            escape_routes: 2,
            future_worst_exits: 3,
            future_escape_routes: 2,
            enemy_cutoff_risk: 0,
            enemy_push_risk: 0,
            edge_exposure_risk: 0,
            commitment_risk: 0,
            loses_head_to_head: false,
        };

        assert!(!is_cerco_risk(&candidate));
    }

    #[test]
    fn boundary_exposure_is_zero_on_wrapped_maps() {
        let board = board(5, 5, &[], &[], vec![]);
        assert_eq!(
            boundary_exposure_risk(&board, Point { x: 0, y: 2 }, 1, 500, true),
            0
        );
    }

    #[test]
    fn historical_turn_180_keeps_the_safe_tail_chase() {
        let me = snake(
            "me",
            96,
            &[
                (6, 0),
                (7, 0),
                (8, 0),
                (9, 0),
                (10, 0),
                (10, 1),
                (10, 2),
                (10, 3),
                (10, 4),
                (10, 5),
                (10, 6),
                (10, 7),
                (10, 8),
                (9, 8),
                (9, 7),
                (9, 6),
                (9, 5),
                (9, 4),
                (9, 3),
                (9, 2),
                (9, 1),
                (8, 1),
                (7, 1),
                (6, 1),
            ],
        );
        let enemy = snake("enemy", 16, &[(6, 8), (6, 9), (5, 9), (5, 10)]);
        let board = board(
            11,
            11,
            &[(0, 0), (2, 1), (0, 9), (1, 7)],
            &[],
            vec![
                snake(
                    "me",
                    96,
                    &[
                        (6, 0),
                        (7, 0),
                        (8, 0),
                        (9, 0),
                        (10, 0),
                        (10, 1),
                        (10, 2),
                        (10, 3),
                        (10, 4),
                        (10, 5),
                        (10, 6),
                        (10, 7),
                        (10, 8),
                        (9, 8),
                        (9, 7),
                        (9, 6),
                        (9, 5),
                        (9, 4),
                        (9, 3),
                        (9, 2),
                        (9, 1),
                        (8, 1),
                        (7, 1),
                        (6, 1),
                    ],
                ),
                enemy,
            ],
        );

        assert_eq!(
            select_direction(&game("standard", None), &board, &me),
            Direction::Up
        );
    }
}
