//! V3.3-Stable: a survival-first Battlesnake with local geometry, food discipline,
//! and bounded adversarial search.
//!
//! V2.8 adds shared iterative deepening and an exact transposition cache so the
//! engine spends its move budget across all root choices instead of timing out
//! on the first branch it explores.
//!
//! V2.8-Food adds starvation-aware route planning: once health starts trending
//! toward the danger zone, the engine explicitly prefers moves that make real
//! progress toward reachable food instead of treating food as a small score
//! bonus below space/escape heuristics.
//!
//! The priority order is: avoid certain death, avoid losing head-to-heads, avoid
//! traps, preserve future manoeuvring room, account for enemy pressure, then use
//! territory and food to improve the position. The strategic search is bounded
//! and deterministic so bad decisions remain reproducible from replays.

use log::{debug, info, warn};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::TryFrom;
use std::time::{Duration, Instant};

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

// V2.4: future escape preservation. When at least one candidate leaves
// a real escape route after the enemy response, prefer that pool over
// candidates whose projected continuation has no escape route.
const MIN_FUTURE_ESCAPE_PREFERENCE: i64 = 1;

// V2.8: shared iterative deepening. Instead of spending almost the whole
// request budget on one candidate and timing out before comparing the others,
// the deep engine now searches every root move at increasing depths and only
// accepts a depth when all roots finished it. The last complete iteration is
// always the result used for the move decision.
const DEEP_PLY: usize = 12;
const DEEP_BRANCH_WIDTH: usize = 3;
const DEEP_MAX_NODES: u64 = 1_500_000;
const DEEP_MAX_TEST_NODES: u64 = 12_000_000;
const DEEP_RESPONSE_RESERVE_MS: u128 = 65;
const DEEP_ITER_DEPTHS: [usize; 5] = [4, 6, 8, 10, 12];
const DEEP_SAFE_MIN_EXITS: i64 = 2;
const DEEP_SAFE_MIN_ESCAPE: i64 = 1;
const DEEP_LOSS_SCORE: i64 = -1_000_000_000;
const DEEP_WIN_SCORE: i64 = 1_000_000_000;

// V2.6 offensive evaluation: once our own safety floor is intact, prefer
// positions that reduce the opponent's reachable area and mobility.
const DEEP_ENEMY_SPACE_WEIGHT: i64 = 90;
const DEEP_ENEMY_EXIT_WEIGHT: i64 = 6_000;
const DEEP_ENEMY_ESCAPE_WEIGHT: i64 = 3_000;
const DEEP_ENEMY_TRAP_BONUS: i64 = 28_000;

// V2.7: make the deep search care about sustained enemy mobility collapse,
// not only the enemy's mobility on the final leaf. This rewards lines that
// progressively squeeze the opponent while our own safety floor remains intact.
const DEEP_ENEMY_SPACE_COLLAPSE_WEIGHT: i64 = 180;
const DEEP_ENEMY_EXIT_COLLAPSE_WEIGHT: i64 = 9_000;
const DEEP_ENEMY_ESCAPE_COLLAPSE_WEIGHT: i64 = 5_000;
const DEEP_ENEMY_BOX_BONUS: i64 = 45_000;

const FOOD_RESCUE_HEALTH: i32 = 40;
const FOOD_RESCUE_WEIGHT: i64 = 180;
const FOOD_HUNT_TRIGGER: i32 = 70;
const FOOD_ROUTE_MARGIN: i32 = 4;
const FOOD_ROUTE_BONUS: i64 = 75_000;
const FOOD_ROUTE_DISTANCE_WEIGHT: i64 = 8_500;
const FOOD_ROUTE_URGENCY_WEIGHT: i64 = 4_500;
const FOOD_ROUTE_LATE_PENALTY: i64 = 125_000;
const FOOD_ROUTE_LATE_STEP_PENALTY: i64 = 18_000;
const FOOD_ROUTE_NO_PATH_PENALTY: i64 = 180_000;
const FOOD_SAFE_EAT_MIN_EXITS: i64 = 2;
const FOOD_SAFE_EAT_MIN_FUTURE: i64 = 2_500;
const INTERIOR_PREFERENCE_MIN_EDGE: i64 = 2;
const INTERIOR_PREFERENCE_MIN_FUTURE: i64 = 3_000;
const EDGE_COMMITMENT_PENALTY: i64 = 11_000;
const EDGE_ZERO_PENALTY: i64 = 24_000;
const EDGE_RECOVERY_BONUS: i64 = 2_000;

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
    food_distance: Option<i32>,
    food_is_contested: bool,
    food_viable: bool,
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

/// Score how urgently a candidate should move toward the nearest reachable food.
///
/// This is intentionally piecewise rather than a small linear food bonus. A
/// route that reaches food before starvation is a survival resource; a route
/// that reaches food too late should be treated as a failing plan even when it
/// leaves a large amount of open space along the way.
fn food_route_score(
    health_after: i32,
    food_distance: Option<i32>,
    food_is_contested: bool,
    eating: bool,
    constrictor: bool,
) -> i64 {
    if constrictor {
        return 0;
    }

    if eating {
        return if health_after <= FOOD_HUNT_TRIGGER {
            FOOD_ROUTE_BONUS
                + (FOOD_HUNT_TRIGGER - health_after).max(0) as i64
                    * FOOD_ROUTE_URGENCY_WEIGHT
        } else {
            0
        };
    }

    if health_after > FOOD_HUNT_TRIGGER {
        return 0;
    }

    let Some(distance) = food_distance else {
        return -if health_after <= FOOD_RESCUE_HEALTH {
            FOOD_ROUTE_NO_PATH_PENALTY * 2
        } else {
            FOOD_ROUTE_NO_PATH_PENALTY
        };
    };

    let eta = distance.saturating_add(1);
    let margin = health_after - eta;
    let urgency = (FOOD_HUNT_TRIGGER - health_after).max(0) as i64;
    let contest_penalty = if food_is_contested { 30_000 } else { 0 };

    if margin >= FOOD_ROUTE_MARGIN {
        FOOD_ROUTE_BONUS
            + urgency * FOOD_ROUTE_URGENCY_WEIGHT
            - distance as i64 * FOOD_ROUTE_DISTANCE_WEIGHT
            - contest_penalty
    } else if margin >= 0 {
        FOOD_ROUTE_BONUS / 2
            + urgency * FOOD_ROUTE_URGENCY_WEIGHT
            - distance as i64 * FOOD_ROUTE_DISTANCE_WEIGHT
            - contest_penalty
            - (FOOD_ROUTE_MARGIN - margin) as i64 * FOOD_ROUTE_LATE_PENALTY / 4
    } else {
        -FOOD_ROUTE_LATE_PENALTY
            - urgency * FOOD_ROUTE_URGENCY_WEIGHT
            - distance as i64 * FOOD_ROUTE_DISTANCE_WEIGHT
            - (-margin) as i64 * FOOD_ROUTE_LATE_STEP_PENALTY
            - contest_penalty
    }
}

fn candidate_food_priority(candidate: &Candidate) -> bool {
    candidate.eating || candidate.food_distance.is_some()
}

fn prefer_food_routes<'a>(
    pool: Vec<&'a Candidate>,
    health: i32,
) -> (Vec<&'a Candidate>, bool) {
    if health > FOOD_HUNT_TRIGGER {
        return (pool, false);
    }

    let routed: Vec<&Candidate> = pool
        .iter()
        .copied()
        .filter(|candidate| candidate_food_priority(candidate))
        .collect();

    if routed.is_empty() {
        return (pool, false);
    }

    let viable: Vec<&Candidate> = routed
        .iter()
        .copied()
        .filter(|candidate| {
            candidate.eating
                || (candidate.food_viable
                    && candidate.future_survival > 0
                    && candidate.exits >= FOOD_SAFE_EAT_MIN_EXITS
                    && candidate.future_worst_exits >= 2)
        })
        .collect();

    if !viable.is_empty() {
        return (viable, true);
    }

    if health <= FOOD_RESCUE_HEALTH {
        let desperate: Vec<&Candidate> = routed
            .iter()
            .copied()
            .filter(|candidate| {
                candidate.food_distance.is_some()
                    && candidate.future_survival > 0
                    && (candidate.eating || candidate.exits >= 1)
            })
            .collect();
        if !desperate.is_empty() {
            return (desperate, true);
        }
    }

    (pool, false)
}

fn safe_food_candidate(candidate: &Candidate) -> bool {
    !candidate.loses_head_to_head
        && !candidate.food_is_contested
        && candidate.eating
        && candidate.exits >= FOOD_SAFE_EAT_MIN_EXITS
        && candidate.future_survival >= FOOD_SAFE_EAT_MIN_FUTURE
        && candidate.future_worst_exits >= 2
}

fn candidate_target_edge(candidate: &Candidate, you: &Battlesnake, board: &Board, wraps: bool) -> i64 {
    if wraps {
        return 99;
    }
    let head = Point::from(&you.head);
    candidate
        .direction
        .next(head, board, wraps)
        .map(|point| boundary_distance(point, board))
        .unwrap_or(0)
}


/// Select the highest-scoring survivable move.
///
/// `Candidate` only contains moves that do not immediately leave the board,
/// hit a body, or run out of health. A losing head-to-head is retained as a
/// last resort, but is never selected when another survivable move exists.
fn select_direction(game: &Game, board: &Board, you: &Battlesnake) -> Direction {
    let timeout_ms = u128::from(game.timeout);
    let wraps = is_wrapped(game);

    let candidates: Vec<Candidate> = DIRECTIONS
        .iter()
        .filter_map(|direction| evaluate_move(*direction, game, board, you))
        .collect();

    if candidates.is_empty() {
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
        return DIRECTIONS
            .iter()
            .copied()
            .find(|direction| direction.next(Point::from(&you.head), board, wraps).is_some())
            .unwrap_or(Direction::Up);
    }

    let non_losing: Vec<&Candidate> = candidates
        .iter()
        .filter(|candidate| !candidate.loses_head_to_head)
        .collect();
    let mut pool: Vec<&Candidate> = if non_losing.is_empty() {
        candidates.iter().collect()
    } else {
        non_losing
    };

    // Safe food is never casually discarded. This is deliberately ahead of
    // territory/attack logic, because growth is what lets the snake compete
    // instead of spending the whole match at shrinking size.
    let safe_food: Vec<&Candidate> = pool
        .iter()
        .copied()
        .filter(|candidate| safe_food_candidate(candidate))
        .collect();

    if !safe_food.is_empty() {
        pool = safe_food;
    } else {
        let future_viable: Vec<&Candidate> = pool
            .iter()
            .copied()
            .filter(|candidate| candidate.future_survival > 0)
            .collect();
        if !future_viable.is_empty() {
            pool = future_viable;
        }

        let (food_pool, food_hunt_active) = prefer_food_routes(pool.clone(), you.health);
        if food_hunt_active {
            pool = food_pool;
        }

        // Anti-wall preference: if an interior move has room and a real future,
        // do not keep hugging the boundary merely because its territory score
        // is a little higher. No absolute edge ban when that is the only route.
        if !wraps && pool.len() > 1 {
            let interior: Vec<&Candidate> = pool
                .iter()
                .copied()
                .filter(|candidate| {
                    candidate_target_edge(candidate, you, board, wraps) >= INTERIOR_PREFERENCE_MIN_EDGE
                        && candidate.exits >= 2
                        && candidate.future_survival >= INTERIOR_PREFERENCE_MIN_FUTURE
                })
                .collect();
            if !interior.is_empty() {
                pool = interior;
            }
        }

        let escape_safe: Vec<&Candidate> = pool
            .iter()
            .copied()
            .filter(|candidate| candidate.forced_kill || candidate.future_escape_routes >= 1)
            .collect();
        if !escape_safe.is_empty() {
            pool = escape_safe;
        }
    }

    if you.health <= FOOD_RESCUE_HEALTH {
        let immediate_food: Vec<&Candidate> = pool
            .iter()
            .copied()
            .filter(|candidate| {
                candidate.eating && !candidate.loses_head_to_head && candidate.future_survival > 0
            })
            .collect();
        if !immediate_food.is_empty() {
            return finish_selection(game, immediate_food);
        }
    }

    let reserve_ms = DEEP_RESPONSE_RESERVE_MS.min(timeout_ms.saturating_sub(20));
    let deep_budget_ms = timeout_ms.saturating_sub(reserve_ms).max(20);
    let deep_deadline = Instant::now() + Duration::from_millis(deep_budget_ms as u64);
    deep_select_direction(game, board, you, pool, deep_deadline)
}

fn prefer_future_escape<'a>(pool: Vec<&'a Candidate>) -> Vec<&'a Candidate> {
    let preferred: Vec<&Candidate> = pool
        .iter()
        .copied()
        .filter(|candidate| {
            candidate.forced_kill
                || candidate.future_escape_routes >= MIN_FUTURE_ESCAPE_PREFERENCE
        })
        .collect();

    if preferred.is_empty() {
        pool
    } else {
        preferred
    }
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
        "MOVE_DECISION id={} direction={} score={} health_after={} space={} territory={} exits={} eating={} food_distance={:?} food_contested={} food_viable={} forced_kill={} h2h_risk={} future={} trap={} pressure={} adversarial_space={} future_space={} worst_space={} worst_exits={} escape_routes={} future_worst_exits={} future_escape_routes={} cutoff={} push={} edge_risk={} commitment={}",
        game.id,
        best.direction.name(),
        best.score,
        best.health_after,
        best.space,
        best.territory,
        best.exits,
        best.eating,
        best.food_distance,
        best.food_is_contested,
        best.food_viable,
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


#[derive(Clone, Hash, PartialEq, Eq)]
struct DeepState {
    our_body: Vec<Point>,
    our_health: i32,
    enemy_body: Vec<Point>,
    enemy_health: i32,
    eaten_food: u128,
    our_alive: bool,
    enemy_alive: bool,
}

#[derive(Clone, Copy, Debug)]
struct DeepEval {
    score: i64,
    survival_plies: usize,
    min_exits: i64,
    min_escape_routes: i64,
    min_space: usize,
    min_enemy_space: usize,
    min_enemy_exits: i64,
    min_enemy_escape: i64,
}

#[derive(Clone, Copy, Debug)]
struct DeepPrediction {
    score: i64,
    survival_plies: usize,
    min_exits: i64,
    min_escape_routes: i64,
    min_space: usize,
    min_enemy_space: usize,
    min_enemy_exits: i64,
    min_enemy_escape: i64,
    nodes: u64,
    timed_out: bool,
}

struct DeepSearchControl {
    deadline: Instant,
    nodes: u64,
    tt_hits: u64,
    timed_out: bool,
    max_nodes: u64,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct DeepTTKey {
    state: DeepState,
    actor: DeepActor,
    depth_remaining: usize,
    enemy_context: u64,
    baseline_enemy_space: usize,
    baseline_enemy_exits: i64,
    baseline_enemy_escape: i64,
}

#[derive(Clone, Copy)]
struct DeepTTEntry {
    eval: DeepEval,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum DeepActor {
    OurSnake,
    Enemy,
}

fn deep_select_direction(
    game: &Game,
    board: &Board,
    you: &Battlesnake,
    pool: Vec<&Candidate>,
    deadline: Instant,
) -> Direction {
    if pool.len() <= 1 {
        return finish_selection(game, pool);
    }

    // V2.8: one global search budget for the whole root. Every candidate is
    // revisited at progressively deeper horizons. A depth is committed only
    // when every root candidate completed it; this prevents a partial branch
    // from beating an unexplored move.
    let mut control = DeepSearchControl {
        deadline,
        nodes: 0,
        tt_hits: 0,
        timed_out: false,
        max_nodes: DEEP_MAX_NODES,
    };

    let mut tt_tables: Vec<HashMap<DeepTTKey, DeepTTEntry>> =
        (0..pool.len()).map(|_| HashMap::new()).collect();
    let mut completed_predictions: Vec<DeepPrediction> = Vec::new();
    let mut completed_depth = 0usize;

    for depth in DEEP_ITER_DEPTHS {
        if depth > DEEP_PLY || Instant::now() >= deadline || control.nodes >= control.max_nodes {
            break;
        }

        let mut iteration: Vec<DeepPrediction> = Vec::with_capacity(pool.len());
        let mut iteration_complete = true;

        for (index, candidate) in pool.iter().enumerate() {
            if Instant::now() >= deadline || control.nodes >= control.max_nodes {
                iteration_complete = false;
                control.timed_out = true;
                break;
            }

            let before_nodes = control.nodes;
            let prediction = deep_prediction_for_candidate_shared(
                game,
                board,
                you,
                candidate,
                depth,
                &mut control,
                &mut tt_tables[index],
            );
            let candidate_nodes = control.nodes.saturating_sub(before_nodes);

            info!(
                "DEEP_ITER_CANDIDATE id={} depth={} direction={} score={} survival_plies={} min_exits={} min_escape_routes={} min_space={} min_enemy_space={} min_enemy_exits={} min_enemy_escape={} nodes={} total_nodes={} tt_hits={} timed_out={}",
                game.id,
                depth,
                candidate.direction.name(),
                prediction.score,
                prediction.survival_plies,
                prediction.min_exits,
                prediction.min_escape_routes,
                prediction.min_space,
                prediction.min_enemy_space,
                prediction.min_enemy_exits,
                prediction.min_enemy_escape,
                candidate_nodes,
                control.nodes,
                control.tt_hits,
                prediction.timed_out
            );

            if prediction.timed_out || control.timed_out {
                iteration_complete = false;
                break;
            }
            iteration.push(prediction);
        }

        if !iteration_complete || iteration.len() != pool.len() {
            break;
        }

        completed_predictions = iteration;
        completed_depth = depth;
        control.timed_out = false;
    }

    if completed_predictions.len() != pool.len() {
        info!(
            "DEEP_ITER_FALLBACK id={} completed_depth={} total_nodes={}",
            game.id,
            completed_depth,
            control.nodes
        );
        return finish_selection(game, pool);
    }

    // Prefer a line that completed the deepest shared iteration and still
    // preserves our survival floor. Among equally safe lines, attack by
    // minimizing the enemy's projected mobility.
    let mut best_index = 0usize;
    for index in 1..pool.len() {
        let candidate = &completed_predictions[index];
        let best = &completed_predictions[best_index];

        let candidate_safe = candidate.survival_plies >= completed_depth
            && candidate.min_exits >= DEEP_SAFE_MIN_EXITS
            && candidate.min_escape_routes >= DEEP_SAFE_MIN_ESCAPE;
        let best_safe = best.survival_plies >= completed_depth
            && best.min_exits >= DEEP_SAFE_MIN_EXITS
            && best.min_escape_routes >= DEEP_SAFE_MIN_ESCAPE;

        let candidate_food = if safe_food_candidate(pool[index]) {
            2
        } else if you.health <= FOOD_HUNT_TRIGGER && pool[index].food_viable && !pool[index].food_is_contested {
            1
        } else {
            0
        };
        let best_food = if safe_food_candidate(pool[best_index]) {
            2
        } else if you.health <= FOOD_HUNT_TRIGGER && pool[best_index].food_viable && !pool[best_index].food_is_contested {
            1
        } else {
            0
        };

        let candidate_key = (
            candidate_safe,
            candidate_food,
            candidate.min_escape_routes,
            candidate.min_exits,
            candidate.min_space.min(200),
            -(candidate.min_enemy_escape),
            -(candidate.min_enemy_exits),
            -(candidate.min_enemy_space as i64),
            pool[index].score,
        );
        let best_key = (
            best_safe,
            best_food,
            best.min_escape_routes,
            best.min_exits,
            best.min_space.min(200),
            -(best.min_enemy_escape),
            -(best.min_enemy_exits),
            -(best.min_enemy_space as i64),
            pool[best_index].score,
        );

        if candidate_key > best_key {
            best_index = index;
        }
    }

    let chosen_candidate = pool[best_index];
    let chosen = completed_predictions[best_index];
    let mode = if chosen.survival_plies >= completed_depth
        && chosen.min_exits >= DEEP_SAFE_MIN_EXITS
        && chosen.min_escape_routes >= DEEP_SAFE_MIN_ESCAPE
    {
        "iterative_safe"
    } else {
        "iterative_deepest"
    };

    info!(
        "DEEP_DECISION id={} direction={} mode={} depth={} survival_plies={} min_exits={} min_escape_routes={} min_space={} min_enemy_space={} min_enemy_exits={} min_enemy_escape={} total_nodes={} tt_hits={} timed_out={}",
        game.id,
        chosen_candidate.direction.name(),
        mode,
        completed_depth,
        chosen.survival_plies,
        chosen.min_exits,
        chosen.min_escape_routes,
        chosen.min_space,
        chosen.min_enemy_space,
        chosen.min_enemy_exits,
        chosen.min_enemy_escape,
        control.nodes,
        control.tt_hits,
        control.timed_out
    );

    chosen_candidate.direction
}

fn deep_prediction_for_candidate_shared(
    game: &Game,
    board: &Board,
    you: &Battlesnake,
    candidate: &Candidate,
    depth: usize,
    control: &mut DeepSearchControl,
    tt: &mut HashMap<DeepTTKey, DeepTTEntry>,
) -> DeepPrediction {
    let wraps = is_wrapped(game);
    let food = point_set(&board.food);
    let hazards = hazard_counts(&board.hazards);
    let dangerous = larger_head_territory(
        board,
        you,
        snake_length(you) + i32::from(candidate.eating),
        wraps,
    );
    let target = candidate_target(you, candidate.direction, board, wraps)
        .unwrap_or_else(|| Point::from(&you.head));
    let grows = candidate.eating || is_constrictor(game);
    let our_body = projected_body(you, target, grows);

    let mut worst: Option<DeepPrediction> = None;
    let mut total_nodes = 0u64;

    for enemy in board.snakes.iter().filter(|snake| snake.id != you.id) {
        if Instant::now() >= control.deadline || control.nodes >= control.max_nodes {
            control.timed_out = true;
            break;
        }

        let state = DeepState {
            our_body: our_body.clone(),
            our_health: candidate.health_after,
            enemy_body: enemy.body.iter().map(Point::from).collect(),
            enemy_health: enemy.health,
            eaten_food: if candidate.eating {
                food_mask(target, &board.food)
            } else {
                0
            },
            our_alive: true,
            enemy_alive: true,
        };

        let static_blocked: HashSet<Point> = board
            .snakes
            .iter()
            .filter(|snake| snake.id != you.id && snake.id != enemy.id)
            .flat_map(|snake| snake.body.iter().map(Point::from))
            .collect();

        let (baseline_enemy_space, baseline_enemy_exits, baseline_enemy_escape) =
            deep_enemy_mobility(board, &state, wraps);
        let enemy_context = stable_hash(enemy.id.as_bytes());

        let before_nodes = control.nodes;
        let result = deep_minimax(
            game,
            board,
            you,
            &food,
            &hazards,
            &dangerous,
            &static_blocked,
            wraps,
            &state,
            DeepActor::Enemy,
            depth,
            0,
            control,
            DEEP_LOSS_SCORE,
            DEEP_WIN_SCORE,
            baseline_enemy_space,
            baseline_enemy_exits,
            baseline_enemy_escape,
            enemy_context,
            tt,
        );
        total_nodes += control.nodes.saturating_sub(before_nodes);

        if control.timed_out {
            break;
        }

        let prediction = DeepPrediction {
            score: result.score,
            survival_plies: result.survival_plies,
            min_exits: result.min_exits,
            min_escape_routes: result.min_escape_routes,
            min_space: result.min_space,
            min_enemy_space: result.min_enemy_space,
            min_enemy_exits: result.min_enemy_exits,
            min_enemy_escape: result.min_enemy_escape,
            nodes: total_nodes,
            timed_out: false,
        };

        let should_replace = worst.is_none_or(|current: DeepPrediction| {
            (
                prediction.score,
                prediction.survival_plies,
                prediction.min_escape_routes,
                prediction.min_exits,
                -(prediction.min_enemy_escape),
                -(prediction.min_enemy_exits),
                -(prediction.min_enemy_space as i64),
            ) < (
                current.score,
                current.survival_plies,
                current.min_escape_routes,
                current.min_exits,
                -(current.min_enemy_escape),
                -(current.min_enemy_exits),
                -(current.min_enemy_space as i64),
            )
        });
        if should_replace {
            worst = Some(prediction);
        }
    }

    worst.unwrap_or(DeepPrediction {
        score: 0,
        survival_plies: depth,
        min_exits: 4,
        min_escape_routes: 2,
        min_space: board.width.max(0) as usize * board.height.max(0) as usize,
        min_enemy_space: board.width.max(0) as usize * board.height.max(0) as usize,
        min_enemy_exits: 4,
        min_enemy_escape: 2,
        nodes: total_nodes,
        timed_out: control.timed_out,
    })
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn deep_prediction_for_candidate(
    game: &Game,
    board: &Board,
    you: &Battlesnake,
    candidate: &Candidate,
    deadline: Instant,
) -> DeepPrediction {
    deep_prediction_for_candidate_with_limits(
        game,
        board,
        you,
        candidate,
        deadline,
        DEEP_MAX_NODES,
    )
}

fn deep_prediction_for_candidate_with_limits(
    game: &Game,
    board: &Board,
    you: &Battlesnake,
    candidate: &Candidate,
    deadline: Instant,
    max_nodes: u64,
) -> DeepPrediction {
    deep_prediction_for_candidate_with_limits_and_depth(
        game,
        board,
        you,
        candidate,
        deadline,
        max_nodes,
        DEEP_PLY,
    )
}

fn deep_prediction_for_candidate_with_limits_and_depth(
    game: &Game,
    board: &Board,
    you: &Battlesnake,
    candidate: &Candidate,
    deadline: Instant,
    max_nodes: u64,
    depth: usize,
) -> DeepPrediction {
    let wraps = is_wrapped(game);
    let food = point_set(&board.food);
    let hazards = hazard_counts(&board.hazards);
    let dangerous = larger_head_territory(
        board,
        you,
        snake_length(you) + i32::from(candidate.eating),
        wraps,
    );
    let target = candidate_target(you, candidate.direction, board, wraps)
        .unwrap_or_else(|| Point::from(&you.head));
    let grows = candidate.eating || is_constrictor(game);
    let our_body = projected_body(you, target, grows);

    let mut worst: Option<DeepPrediction> = None;
    let mut total_nodes = 0_u64;
    let mut timed_out = false;

    for enemy in board.snakes.iter().filter(|snake| snake.id != you.id) {
        if Instant::now() >= deadline {
            timed_out = true;
            break;
        }

        let state = DeepState {
            our_body: our_body.clone(),
            our_health: candidate.health_after,
            enemy_body: enemy.body.iter().map(Point::from).collect(),
            enemy_health: enemy.health,
            eaten_food: if candidate.eating {
                food_mask(target, &board.food)
            } else {
                0
            },
            our_alive: true,
            enemy_alive: true,
        };

        let static_blocked: HashSet<Point> = board
            .snakes
            .iter()
            .filter(|snake| snake.id != you.id && snake.id != enemy.id)
            .flat_map(|snake| snake.body.iter().map(Point::from))
            .collect();

        let (baseline_enemy_space, baseline_enemy_exits, baseline_enemy_escape) =
            deep_enemy_mobility(board, &state, wraps);

        let mut control = DeepSearchControl {
            deadline,
            nodes: 0,
            tt_hits: 0,
            timed_out: false,
            max_nodes,
        };
        let enemy_context = stable_hash(enemy.id.as_bytes());
        let mut tt = HashMap::new();
        let result = deep_minimax(
            game,
            board,
            you,
            &food,
            &hazards,
            &dangerous,
            &static_blocked,
            wraps,
            &state,
            DeepActor::Enemy,
            depth,
            0,
            &mut control,
            DEEP_LOSS_SCORE,
            DEEP_WIN_SCORE,
            baseline_enemy_space,
            baseline_enemy_exits,
            baseline_enemy_escape,
            enemy_context,
            &mut tt,
        );
        total_nodes += control.nodes;
        timed_out |= control.timed_out;

        let prediction = DeepPrediction {
            score: result.score,
            survival_plies: result.survival_plies,
            min_exits: result.min_exits,
            min_escape_routes: result.min_escape_routes,
            min_space: result.min_space,
            min_enemy_space: result.min_enemy_space,
            min_enemy_exits: result.min_enemy_exits,
            min_enemy_escape: result.min_enemy_escape,
            nodes: control.nodes,
            timed_out: control.timed_out,
        };

        let should_replace = worst.is_none_or(|current: DeepPrediction| {
            (
                prediction.score,
                prediction.survival_plies,
                prediction.min_escape_routes,
                prediction.min_exits,
                -(prediction.min_enemy_escape),
                -(prediction.min_enemy_exits),
                -(prediction.min_enemy_space as i64),
            ) < (
                current.score,
                current.survival_plies,
                current.min_escape_routes,
                current.min_exits,
                -(current.min_enemy_escape),
                -(current.min_enemy_exits),
                -(current.min_enemy_space as i64),
            )
        });
        if should_replace {
            worst = Some(prediction);
        }
    }

    worst.unwrap_or(DeepPrediction {
        score: 0,
        survival_plies: DEEP_PLY,
        min_exits: 4,
        min_escape_routes: 2,
        min_space: board.width.max(0) as usize * board.height.max(0) as usize,
        min_enemy_space: board.width.max(0) as usize * board.height.max(0) as usize,
        min_enemy_exits: 4,
        min_enemy_escape: 2,
        nodes: total_nodes,
        timed_out,
    })
}

fn candidate_target(
    you: &Battlesnake,
    direction: Direction,
    board: &Board,
    wraps: bool,
) -> Option<Point> {
    direction.next(Point::from(&you.head), board, wraps)
}

fn deep_tt_store(
    tt: &mut HashMap<DeepTTKey, DeepTTEntry>,
    key: &DeepTTKey,
    eval: DeepEval,
    depth_from_root: usize,
) {
    let mut cached = eval;
    // Store survival relative to this node so the same exact state can be
    // reused by iterative-deepening passes even when it is reached at a
    // different absolute depth from the root.
    cached.survival_plies = cached.survival_plies.saturating_sub(depth_from_root);
    tt.insert(key.clone(), DeepTTEntry { eval: cached });
}

fn deep_minimax(
    game: &Game,
    board: &Board,
    you: &Battlesnake,
    food: &HashSet<Point>,
    hazards: &HashMap<Point, i32>,
    dangerous: &HashSet<Point>,
    static_blocked: &HashSet<Point>,
    wraps: bool,
    state: &DeepState,
    actor: DeepActor,
    depth_remaining: usize,
    depth_from_root: usize,
    control: &mut DeepSearchControl,
    mut alpha: i64,
    mut beta: i64,
    baseline_enemy_space: usize,
    baseline_enemy_exits: i64,
    baseline_enemy_escape: i64,
    enemy_context: u64,
    tt: &mut HashMap<DeepTTKey, DeepTTEntry>,
) -> DeepEval {
    control.nodes += 1;
    if control.nodes >= control.max_nodes || Instant::now() >= control.deadline {
        control.timed_out = true;
        return deep_eval_state(board, state, food, static_blocked, dangerous, wraps, depth_from_root, baseline_enemy_space, baseline_enemy_exits, baseline_enemy_escape);
    }

    let tt_key = DeepTTKey {
        state: state.clone(),
        actor,
        depth_remaining,
        enemy_context,
        baseline_enemy_space,
        baseline_enemy_exits,
        baseline_enemy_escape,
    };
    if let Some(entry) = tt.get(&tt_key).copied() {
        control.tt_hits += 1;
        let mut eval = entry.eval;
        eval.survival_plies = eval.survival_plies.saturating_add(depth_from_root);
        return eval;
    }

    if !state.our_alive {
        let eval = DeepEval {
            score: DEEP_LOSS_SCORE,
            survival_plies: depth_from_root,
            min_exits: 0,
            min_escape_routes: 0,
            min_space: 0,
            min_enemy_space: 0,
            min_enemy_exits: 0,
            min_enemy_escape: 0,
        };
        deep_tt_store(tt, &tt_key, eval, depth_from_root);
        return eval;
    }

    if depth_remaining == 0 || !state.enemy_alive {
        let eval = deep_eval_state(board, state, food, static_blocked, dangerous, wraps, depth_from_root, baseline_enemy_space, baseline_enemy_exits, baseline_enemy_escape);
        deep_tt_store(tt, &tt_key, eval, depth_from_root);
        return eval;
    }

    let states = deep_generate_moves(
        game,
        board,
        you,
        food,
        hazards,
        dangerous,
        static_blocked,
        wraps,
        state,
        actor,
    );

    if states.is_empty() {
        if matches!(actor, DeepActor::OurSnake) {
            let eval = DeepEval {
                score: DEEP_LOSS_SCORE,
                survival_plies: depth_from_root,
                min_exits: 0,
                min_escape_routes: 0,
                min_space: 0,
                min_enemy_space: 0,
                min_enemy_exits: 0,
                min_enemy_escape: 0,
            };
            deep_tt_store(tt, &tt_key, eval, depth_from_root);
            return eval;
        }

        // If the enemy has no legal move, it dies under Battlesnake rules.
        // Treating this as a neutral leaf was making the offensive search miss
        // genuine squeeze/trap wins.
        let mut dead_enemy = state.clone();
        dead_enemy.enemy_alive = false;
        let eval = deep_eval_state(
            board,
            &dead_enemy,
            food,
            static_blocked,
            dangerous,
            wraps,
            depth_from_root,
            baseline_enemy_space,
            baseline_enemy_exits,
            baseline_enemy_escape,
        );
        deep_tt_store(tt, &tt_key, eval, depth_from_root);
        return eval;
    }

    let mut ranked = states;
    ranked.sort_unstable_by(|a, b| {
        let a_score = deep_quick_state_score(board, a, food, dangerous, wraps, baseline_enemy_space, baseline_enemy_exits, baseline_enemy_escape);
        let b_score = deep_quick_state_score(board, b, food, dangerous, wraps, baseline_enemy_space, baseline_enemy_exits, baseline_enemy_escape);
        match actor {
            DeepActor::OurSnake => b_score.cmp(&a_score),
            DeepActor::Enemy => a_score.cmp(&b_score),
        }
    });
    // V2.7: `ranked` is already ordered by deep_quick_state_score.
    // Search the most relevant branches first so alpha-beta can prune much more
    // aggressively, which is why we can afford 12 plies with only 3 branches.
    ranked.truncate(DEEP_BRANCH_WIDTH);

    let mut best: Option<DeepEval> = None;
    let mut pruned = false;
    for child in ranked {
        let child_eval = deep_minimax(
            game,
            board,
            you,
            food,
            hazards,
            dangerous,
            static_blocked,
            wraps,
            &child,
            match actor {
                DeepActor::OurSnake => DeepActor::Enemy,
                DeepActor::Enemy => DeepActor::OurSnake,
            },
            depth_remaining - 1,
            depth_from_root + 1,
            control,
            alpha,
            beta,
            baseline_enemy_space,
            baseline_enemy_exits,
            baseline_enemy_escape,
            enemy_context,
            tt,
        );

        let (current_exits, current_escape) = deep_quick_escape_metrics(
            board,
            state,
            dangerous,
            wraps,
        );
        let (current_enemy_space, current_enemy_exits, current_enemy_escape) =
            deep_enemy_mobility(board, state, wraps);
        let child_eval = DeepEval {
            score: child_eval.score,
            survival_plies: child_eval.survival_plies.max(depth_from_root + 1),
            min_exits: child_eval.min_exits.min(current_exits),
            min_escape_routes: child_eval.min_escape_routes.min(current_escape),
            min_space: child_eval.min_space,
            min_enemy_space: child_eval.min_enemy_space.min(current_enemy_space),
            min_enemy_exits: child_eval.min_enemy_exits.min(current_enemy_exits),
            min_enemy_escape: child_eval.min_enemy_escape.min(current_enemy_escape),
        };

        let better = best.is_none_or(|previous| match actor {
            DeepActor::OurSnake => deep_eval_is_better_for_us(child_eval, previous),
            DeepActor::Enemy => deep_eval_is_worse_for_us(child_eval, previous),
        });
        if better {
            best = Some(child_eval);
        }

        match actor {
            DeepActor::OurSnake => {
                alpha = alpha.max(child_eval.score);
            }
            DeepActor::Enemy => {
                beta = beta.min(child_eval.score);
            }
        }

        if beta <= alpha || control.timed_out {
            pruned = beta <= alpha;
            break;
        }
    }

    let eval = best.unwrap_or_else(|| deep_eval_state(board, state, food, static_blocked, dangerous, wraps, depth_from_root, baseline_enemy_space, baseline_enemy_exits, baseline_enemy_escape));
    if !pruned && !control.timed_out {
        deep_tt_store(tt, &tt_key, eval, depth_from_root);
    }
    eval
}

fn deep_eval_is_better_for_us(a: DeepEval, b: DeepEval) -> bool {
    (
        a.score,
        a.survival_plies,
        a.min_escape_routes,
        a.min_exits,
        -(a.min_enemy_escape),
        -(a.min_enemy_exits),
        -(a.min_enemy_space as i64),
        a.min_space,
    )
        > (
            b.score,
            b.survival_plies,
            b.min_escape_routes,
            b.min_exits,
            -(b.min_enemy_escape),
            -(b.min_enemy_exits),
            -(b.min_enemy_space as i64),
            b.min_space,
        )
}

fn deep_eval_is_worse_for_us(a: DeepEval, b: DeepEval) -> bool {
    (
        a.score,
        a.survival_plies,
        a.min_escape_routes,
        a.min_exits,
        -(a.min_enemy_escape),
        -(a.min_enemy_exits),
        -(a.min_enemy_space as i64),
        a.min_space,
    )
        < (
            b.score,
            b.survival_plies,
            b.min_escape_routes,
            b.min_exits,
            -(b.min_enemy_escape),
            -(b.min_enemy_exits),
            -(b.min_enemy_space as i64),
            b.min_space,
        )
}

fn deep_generate_moves(
    game: &Game,
    board: &Board,
    _you: &Battlesnake,
    food: &HashSet<Point>,
    hazards: &HashMap<Point, i32>,
    dangerous: &HashSet<Point>,
    static_blocked: &HashSet<Point>,
    wraps: bool,
    state: &DeepState,
    actor: DeepActor,
) -> Vec<DeepState> {
    let (body, health, enemy_body, _enemy_health, our_turn) = match actor {
        DeepActor::OurSnake => (
            &state.our_body,
            state.our_health,
            &state.enemy_body,
            state.enemy_health,
            true,
        ),
        DeepActor::Enemy => (
            &state.enemy_body,
            state.enemy_health,
            &state.our_body,
            state.our_health,
            false,
        ),
    };

    let head = body[0];
    let neck = body.get(1).copied();
    let enemy_head = enemy_body[0];
    let mut result = Vec::new();

    for direction in DIRECTIONS.iter().copied() {
        let Some(target) = direction.next(head, board, wraps) else {
            continue;
        };
        if neck == Some(target) {
            continue;
        }

        let food_here = food.contains(&target)
            && state.eaten_food & food_mask(target, &board.food) == 0;
        let grows = is_constrictor(game) || food_here;
        let own_body_end = if grows {
            body.len()
        } else {
            body.len().saturating_sub(1)
        };
        if body[..own_body_end].contains(&target) {
            continue;
        }

        if static_blocked.contains(&target) {
            continue;
        }

        let mut kills_other = false;
        if target == enemy_head {
            let actor_length = body.len() as i32 + i32::from(grows);
            let enemy_length = enemy_body.len() as i32;
            if actor_length <= enemy_length {
                continue;
            }
            kills_other = true;
        } else if enemy_body.contains(&target) {
            continue;
        }

        // The deep model intentionally treats the enemy as an adversary. The
        // enemy is allowed to choose a move onto our head only when it can win
        // that head-to-head. That becomes an explicit terminal loss for us.
        if !our_turn && target == state.our_body[0] {
            let enemy_length = body.len() as i32 + i32::from(grows);
            let our_length = state.our_body.len() as i32;
            if enemy_length <= our_length {
                continue;
            }
            let mut killed = state.clone();
            killed.our_alive = false;
            killed.enemy_health = if grows { 100 } else { health - 1 };
            result.push(killed);
            continue;
        }

        if our_turn && dangerous.contains(&target) {
            continue;
        }

        let hazard_cost = hazards.get(&target).copied().unwrap_or(0) * hazard_damage(game);
        let next_health = if grows { 100 } else { health - 1 - hazard_cost };
        if next_health <= 0 {
            continue;
        }

        let mut next_body = Vec::with_capacity(body.len() + 1);
        next_body.push(target);
        next_body.extend(body.iter().copied());
        if !grows {
            next_body.pop();
        }

        let mut next = state.clone();
        next.eaten_food |= if food_here {
            food_mask(target, &board.food)
        } else {
            0
        };

        if our_turn {
            next.our_body = next_body;
            next.our_health = next_health;
            if kills_other {
                next.enemy_alive = false;
                next.enemy_body = vec![enemy_head];
            }
        } else {
            next.enemy_body = next_body;
            next.enemy_health = next_health;
            if kills_other {
                next.our_alive = false;
            }
        }

        result.push(next);
    }

    result
}

fn deep_nearest_food_distance(
    board: &Board,
    state: &DeepState,
    food: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    wraps: bool,
) -> Option<i32> {
    if !state.our_alive {
        return None;
    }

    let mut blocked = HashSet::new();
    blocked.extend(state.our_body.iter().copied());
    blocked.extend(state.enemy_body.iter().copied());
    if let Some(tail) = state.our_body.last().copied() {
        blocked.remove(&tail);
    }

    let (_, distances) = reachable_space(
        board,
        state.our_body[0],
        &blocked,
        dangerous,
        wraps,
    );

    food.iter()
        .copied()
        .filter(|point| state.eaten_food & food_mask(*point, &board.food) == 0)
        .filter_map(|point| distances.get(&point).copied())
        .min()
}

fn deep_food_pressure(
    board: &Board,
    state: &DeepState,
    food: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    wraps: bool,
) -> i64 {
    if !state.our_alive {
        return 0;
    }

    if state.our_health > FOOD_HUNT_TRIGGER {
        return 0;
    }

    let distance = deep_nearest_food_distance(board, state, food, dangerous, wraps);
    food_route_score(state.our_health, distance, false, false, false)
}

fn deep_quick_state_score(
    board: &Board,
    state: &DeepState,
    food: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    wraps: bool,
    baseline_enemy_space: usize,
    baseline_enemy_exits: i64,
    baseline_enemy_escape: i64,
) -> i64 {
    if !state.our_alive {
        return DEEP_LOSS_SCORE;
    }

    let (exits, escape) = deep_quick_escape_metrics(board, state, dangerous, wraps);
    let edge = boundary_distance(state.our_body[0], board);
    let enemy_distance = topology_distance(state.our_body[0], state.enemy_body[0], board, wraps);
    let (enemy_space, enemy_exits, enemy_escape) = deep_enemy_mobility(board, state, wraps);
    let our_safe = exits >= 2 && escape >= 1;

    // V2.7: sustained squeeze ordering. During alpha-beta move ordering we want
    // our branches that preserve safety and compress enemy mobility to be searched
    // first. This materially improves pruning without adding another simulation.
    let offensive = if state.enemy_alive && our_safe {
        (board.width.max(0) as i64 * board.height.max(0) as i64 - enemy_space as i64).max(0)
            * DEEP_ENEMY_SPACE_WEIGHT
            + (4 - enemy_exits).max(0) * DEEP_ENEMY_EXIT_WEIGHT
            + (2 - enemy_escape).max(0) * DEEP_ENEMY_ESCAPE_WEIGHT
            + (board.width.max(0) as i64 * board.height.max(0) as i64 - enemy_space as i64).max(0)
                * DEEP_ENEMY_SPACE_COLLAPSE_WEIGHT
            + (4 - enemy_exits).max(0) * DEEP_ENEMY_EXIT_COLLAPSE_WEIGHT
            + (2 - enemy_escape).max(0) * DEEP_ENEMY_ESCAPE_COLLAPSE_WEIGHT
            + (baseline_enemy_space as i64 - enemy_space as i64).max(0)
                * DEEP_ENEMY_SPACE_COLLAPSE_WEIGHT
            + (baseline_enemy_exits - enemy_exits).max(0) * DEEP_ENEMY_EXIT_COLLAPSE_WEIGHT
            + (baseline_enemy_escape - enemy_escape).max(0) * DEEP_ENEMY_ESCAPE_COLLAPSE_WEIGHT
            + if enemy_exits <= 1 && enemy_escape == 0 && enemy_space <= 12 {
                DEEP_ENEMY_BOX_BONUS
            } else {
                0
            }
    } else {
        0
    };

    let food_pressure = deep_food_pressure(board, state, food, dangerous, wraps);

    exits * 18_000
        + escape * 16_000
        + edge * 700
        + state.our_health as i64 * 20
        - enemy_distance * 300
        + offensive
        + food_pressure
        + if state.enemy_alive { 0 } else { 100_000 }
}

fn deep_quick_escape_metrics(
    board: &Board,
    state: &DeepState,
    dangerous: &HashSet<Point>,
    wraps: bool,
) -> (i64, i64) {
    if !state.our_alive {
        return (0, 0);
    }

    let mut blocked = HashSet::new();
    blocked.extend(state.our_body.iter().copied());
    blocked.extend(state.enemy_body.iter().copied());
    if let Some(tail) = state.our_body.last().copied() {
        blocked.remove(&tail);
    }

    let exits = count_escape_exits(board, state.our_body[0], &blocked, dangerous, wraps) as i64;
    let escape = escape_routes_away_from_enemies(
        board,
        state.our_body[0],
        &blocked,
        dangerous,
        state.enemy_body[0],
        wraps,
    ) as i64;
    (exits, escape)
}

fn deep_enemy_mobility(
    board: &Board,
    state: &DeepState,
    wraps: bool,
) -> (usize, i64, i64) {
    if !state.enemy_alive {
        return (0, 0, 0);
    }

    let mut blocked = HashSet::new();
    blocked.extend(state.our_body.iter().copied());
    blocked.extend(state.enemy_body.iter().copied());
    if let Some(tail) = state.enemy_body.last().copied() {
        blocked.remove(&tail);
    }

    let enemy_head = state.enemy_body[0];
    let empty_danger = HashSet::new();
    let exits = count_escape_exits(board, enemy_head, &blocked, &empty_danger, wraps) as i64;
    let escape = escape_routes_away_from_enemies(
        board,
        enemy_head,
        &blocked,
        &empty_danger,
        state.our_body[0],
        wraps,
    ) as i64;
    let (space, _) = reachable_space(board, enemy_head, &blocked, &empty_danger, wraps);
    (space, exits, escape)
}

fn deep_eval_state(
    board: &Board,
    state: &DeepState,
    food: &HashSet<Point>,
    static_blocked: &HashSet<Point>,
    dangerous: &HashSet<Point>,
    wraps: bool,
    depth_from_root: usize,
    baseline_enemy_space: usize,
    baseline_enemy_exits: i64,
    baseline_enemy_escape: i64,
) -> DeepEval {
    if !state.our_alive {
        return DeepEval {
            score: DEEP_LOSS_SCORE + depth_from_root as i64,
            survival_plies: depth_from_root,
            min_exits: 0,
            min_escape_routes: 0,
            min_space: 0,
            min_enemy_space: 0,
            min_enemy_exits: 0,
            min_enemy_escape: 0,
        };
    }

    let mut blocked = static_blocked.clone();
    blocked.extend(state.our_body.iter().copied());
    blocked.extend(state.enemy_body.iter().copied());
    if let Some(tail) = state.our_body.last().copied() {
        blocked.remove(&tail);
    }

    let exits = count_escape_exits(board, state.our_body[0], &blocked, dangerous, wraps) as i64;
    let escape_routes = escape_routes_away_from_enemies(
        board,
        state.our_body[0],
        &blocked,
        dangerous,
        state.enemy_body[0],
        wraps,
    ) as i64;
    let (space, _) = reachable_space(board, state.our_body[0], &blocked, dangerous, wraps);
    let edge = boundary_distance(state.our_body[0], board);
    let (enemy_space, enemy_exits, enemy_escape) = deep_enemy_mobility(board, state, wraps);
    let food_pressure = deep_food_pressure(board, state, food, dangerous, wraps);

    let mut score = space as i64 * 320
        + exits * 18_000
        + escape_routes * 16_000
        + edge * 700
        + state.our_health as i64 * 25;

    if exits <= 1 {
        score -= 45_000;
    }
    if escape_routes == 0 {
        score -= 60_000;
    }
    if edge == 0 && exits <= 2 {
        score -= 25_000;
    }
    if state.enemy_alive {
        let enemy_distance = topology_distance(
            state.our_body[0],
            state.enemy_body[0],
            board,
            wraps,
        );
        score -= enemy_distance.saturating_sub(2) * 180;

        // Offensive objective: once our own position is defensible, make the
        // enemy's world smaller. This turns the deep search into a predator
        // rather than a pure avoid-everything snake.
        if exits >= 2 && escape_routes >= 1 {
            let board_area = board.width.max(0) as i64 * board.height.max(0) as i64;
            score += (board_area - enemy_space as i64).max(0) * DEEP_ENEMY_SPACE_WEIGHT;
            score += (4 - enemy_exits).max(0) * DEEP_ENEMY_EXIT_WEIGHT;
            score += (2 - enemy_escape).max(0) * DEEP_ENEMY_ESCAPE_WEIGHT;

            // V2.7: reward a sustained squeeze more aggressively when the enemy
            // is already down to a narrow mobility envelope.
            score += (board_area - enemy_space as i64).max(0)
                * DEEP_ENEMY_SPACE_COLLAPSE_WEIGHT;
            score += (4 - enemy_exits).max(0) * DEEP_ENEMY_EXIT_COLLAPSE_WEIGHT;
            score += (2 - enemy_escape).max(0) * DEEP_ENEMY_ESCAPE_COLLAPSE_WEIGHT;

            if enemy_exits <= 1 && enemy_space <= 10 {
                score += DEEP_ENEMY_TRAP_BONUS;
            }
            if enemy_exits <= 1 && enemy_escape == 0 && enemy_space <= 12 {
                score += DEEP_ENEMY_BOX_BONUS;
            }

            // V2.7: reward compression relative to the position where this
            // candidate started. This turns the deep search into a real funnel
            // predictor: the enemy does not merely need to be cramped at the
            // leaf; our line should progressively reduce its options.
            let space_collapse = (baseline_enemy_space as i64 - enemy_space as i64).max(0);
            let exit_collapse = (baseline_enemy_exits - enemy_exits).max(0);
            let escape_collapse = (baseline_enemy_escape - enemy_escape).max(0);
            score += space_collapse * DEEP_ENEMY_SPACE_COLLAPSE_WEIGHT;
            score += exit_collapse * DEEP_ENEMY_EXIT_COLLAPSE_WEIGHT;
            score += escape_collapse * DEEP_ENEMY_ESCAPE_COLLAPSE_WEIGHT;
        }
    } else {
        score += 100_000;
    }

    DeepEval {
        score,
        survival_plies: depth_from_root,
        min_exits: exits,
        min_escape_routes: escape_routes,
        min_space: space,
        min_enemy_space: enemy_space,
        min_enemy_exits: enemy_exits,
        min_enemy_escape: enemy_escape,
    }
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

    if !wraps {
        let edge = boundary_distance(target, board);
        if edge == 0 && exits <= 2 {
            score -= EDGE_ZERO_PENALTY;
        } else if edge == 1 && exits <= 2 {
            score -= EDGE_COMMITMENT_PENALTY;
        } else if edge >= INTERIOR_PREFERENCE_MIN_EDGE {
            score += EDGE_RECOVERY_BONUS;
        }
    }

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

    score += food_route_score(
        health_after,
        food_distance,
        food_is_contested,
        eating,
        constrictor,
    );

    if eating
        && !loses_head_to_head
        && !food_is_contested
        && exits >= FOOD_SAFE_EAT_MIN_EXITS
        && future_survival >= FOOD_SAFE_EAT_MIN_FUTURE
    {
        score += 85_000;
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
        food_distance,
        food_is_contested,
        food_viable: food_distance.is_some_and(|distance| {
            let eta = distance.saturating_add(1);
            eta <= health_after.saturating_sub(FOOD_ROUTE_MARGIN)
                && !food_is_contested
        }),
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
        if closest.is_none_or(|(best_distance, best_contested)| {
            (!contested && best_contested)
                || (contested == best_contested && distance < best_distance)
        }) {
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
    fn prefer_food_routes_enters_hunt_mode_below_the_trigger() {
        let food_candidate = Candidate {
            direction: Direction::Right,
            score: -100_000,
            health_after: 55,
            space: 30,
            exits: 2,
            eating: false,
            territory: 10,
            forced_kill: false,
            future_survival: 10_000,
            trap_risk: 0,
            enemy_pressure: 0,
            adversarial_space: 30,
            future_space: 30,
            worst_case_space: 30,
            worst_case_exits: 1,
            escape_routes: 1,
            future_worst_exits: 2,
            future_escape_routes: 0,
            enemy_cutoff_risk: 0,
            enemy_push_risk: 0,
            edge_exposure_risk: 0,
            commitment_risk: 0,
            food_distance: Some(4),
            food_is_contested: false,
            food_viable: true,
            loses_head_to_head: false,
        };
        let spacious = Candidate {
            direction: Direction::Up,
            food_distance: None,
            ..food_candidate
        };

        let (preferred, active) = prefer_food_routes(vec![&spacious, &food_candidate], 55);
        assert!(active);
        assert_eq!(preferred.len(), 1);
        assert_eq!(preferred[0].direction, Direction::Right);
    }

    #[test]
    fn food_route_score_prefers_reachable_food_when_health_is_low() {
        let near = food_route_score(30, Some(3), false, false, false);
        let far = food_route_score(30, Some(12), false, false, false);
        let no_food = food_route_score(30, None, false, false, false);

        assert!(near > far);
        assert!(near > no_food);
    }

    #[test]
    fn food_route_score_treats_late_food_as_a_survival_failure() {
        let reachable = food_route_score(10, Some(2), false, false, false);
        let late = food_route_score(10, Some(15), false, false, false);

        assert!(reachable > late);
        assert!(late < 0);
    }

    #[test]
    fn nearest_food_prefers_uncontested_food_over_a_closer_contested_food() {
        let food = HashSet::from([
            Point { x: 3, y: 2 },
            Point { x: 5, y: 2 },
        ]);
        let mut our_distances = HashMap::new();
        our_distances.insert(Point { x: 3, y: 2 }, 1);
        our_distances.insert(Point { x: 5, y: 2 }, 3);
        let mut enemy_path = HashMap::new();
        enemy_path.insert(Point { x: 3, y: 2 }, 1);
        enemy_path.insert(Point { x: 5, y: 2 }, 99);

        let (distance, contested) = nearest_food(
            &food,
            Point { x: 2, y: 2 },
            4,
            &our_distances,
            &[(4, enemy_path)],
        );

        assert_eq!(distance, Some(3));
        assert!(!contested);
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
    fn deep_prediction_reaches_multiple_alternating_plies() {
        let me = snake("me", 100, &[(1, 1), (1, 0), (0, 0)]);
        let enemy = snake("enemy", 100, &[(8, 8), (8, 9), (9, 9)]);
        let board = board(11, 11, &[], &[], vec![snake("me", 100, &[(1, 1), (1, 0), (0, 0)]), enemy]);
        let game = game("standard", None);
        let candidate = evaluate_move(Direction::Up, &game, &board, &me)
            .expect("up should be a legal candidate on an open board");

        // The production search is intentionally time-bounded. This unit test
        // needs a generous deterministic budget so it verifies the full
        // multi-ply search rather than merely verifying the timeout fallback.
        // The production search is 12 plies deep. This unit test only needs to
        // prove the alternating adversarial recursion itself, so it uses a
        // smaller deterministic horizon to avoid making the test dependent on
        // the machine's raw CPU speed.
        const DEEP_TEST_PLY: usize = 6;
        let prediction = deep_prediction_for_candidate_with_limits_and_depth(
            &game,
            &board,
            &me,
            &candidate,
            Instant::now() + Duration::from_secs(5),
            DEEP_MAX_TEST_NODES,
            DEEP_TEST_PLY,
        );

        assert!(prediction.nodes > 0);
        assert!(!prediction.timed_out);
        assert!(prediction.survival_plies >= DEEP_TEST_PLY);
        assert!(prediction.min_exits >= 2);
        assert!(prediction.min_escape_routes >= 1);
    }

    #[test]
    fn deep_state_score_rewards_a_safe_enemy_mobility_collapse() {
        let me = snake("me", 100, &[(2, 2), (2, 1), (2, 0)]);
        let enemy_open = snake("enemy", 100, &[(8, 8), (8, 9), (9, 9)]);
        let enemy_boxed = snake("enemy", 100, &[(10, 10), (10, 9), (9, 10), (9, 9)]);
        let open_board = board(
            11,
            11,
            &[],
            &[],
            vec![
                snake("me", 100, &[(2, 2), (2, 1), (2, 0)]),
                enemy_open,
            ],
        );
        let boxed_board = board(
            11,
            11,
            &[],
            &[],
            vec![
                snake("me", 100, &[(2, 2), (2, 1), (2, 0)]),
                enemy_boxed,
            ],
        );
        let dangerous = HashSet::new();
        let open_state = DeepState {
            our_body: me.body.iter().map(Point::from).collect(),
            our_health: 100,
            enemy_body: vec![Point { x: 8, y: 8 }, Point { x: 8, y: 9 }, Point { x: 9, y: 9 }],
            enemy_health: 100,
            eaten_food: 0,
            our_alive: true,
            enemy_alive: true,
        };
        let boxed_state = DeepState {
            our_body: me.body.iter().map(Point::from).collect(),
            our_health: 100,
            enemy_body: vec![
                Point { x: 10, y: 10 },
                Point { x: 10, y: 9 },
                Point { x: 9, y: 10 },
                Point { x: 9, y: 9 },
            ],
            enemy_health: 100,
            eaten_food: 0,
            our_alive: true,
            enemy_alive: true,
        };

        let open_score = deep_quick_state_score(&open_board, &open_state, &point_set(&open_board.food), &dangerous, false, 3, 2, 2);
        let boxed_score = deep_quick_state_score(&boxed_board, &boxed_state, &point_set(&boxed_board.food), &dangerous, false, 3, 2, 2);

        assert!(boxed_score > open_score);
    }

    #[test]
    fn deep_prediction_marks_a_forced_head_to_head_loss() {
        let me = snake("me", 100, &[(0, 1), (0, 0)]);
        let enemy = snake("enemy", 100, &[(2, 1), (2, 0), (3, 0), (3, 1)]);
        let board = board(5, 5, &[], &[], vec![snake("me", 100, &[(0, 1), (0, 0)]), enemy]);
        let game = game("standard", None);
        let candidate = Candidate {
            direction: Direction::Right,
            score: 0,
            health_after: 99,
            space: 20,
            exits: 3,
            eating: false,
            territory: 10,
            forced_kill: false,
            future_survival: 10_000,
            trap_risk: 0,
            enemy_pressure: 0,
            adversarial_space: 20,
            future_space: 100,
            worst_case_space: 20,
            worst_case_exits: 3,
            escape_routes: 2,
            future_worst_exits: 3,
            future_escape_routes: 2,
            enemy_cutoff_risk: 0,
            enemy_push_risk: 0,
            edge_exposure_risk: 0,
            commitment_risk: 0,
            food_distance: None,
            food_is_contested: false,
            food_viable: false,
            loses_head_to_head: false,
        };

        let prediction = deep_prediction_for_candidate(
            &game,
            &board,
            &me,
            &candidate,
            Instant::now() + Duration::from_millis(250),
        );

        assert!(prediction.score <= DEEP_LOSS_SCORE + DEEP_PLY as i64);
        assert!(prediction.survival_plies <= 1);
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
            food_distance: None,
            food_is_contested: false,
            food_viable: false,
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
            food_distance: None,
            food_is_contested: false,
            food_viable: false,
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
    fn prefers_a_candidate_with_a_future_escape_route() {
        let trapped = Candidate {
            direction: Direction::Right,
            score: 100_000,
            health_after: 90,
            space: 110,
            exits: 2,
            eating: false,
            territory: 50,
            forced_kill: false,
            future_survival: 18_000,
            trap_risk: 0,
            enemy_pressure: 0,
            adversarial_space: 110,
            future_space: 120,
            worst_case_space: 110,
            worst_case_exits: 2,
            escape_routes: 1,
            future_worst_exits: 2,
            future_escape_routes: 0,
            enemy_cutoff_risk: 1_000,
            enemy_push_risk: 2_900,
            edge_exposure_risk: 1_200,
            commitment_risk: 6_600,
            food_distance: None,
            food_is_contested: false,
            food_viable: false,
            loses_head_to_head: false,
        };
        let safe = Candidate {
            direction: Direction::Up,
            score: -100_000,
            future_escape_routes: 1,
            ..trapped
        };

        let pool = vec![&trapped, &safe];
        let preferred = prefer_future_escape(pool);

        assert_eq!(preferred.len(), 1);
        assert_eq!(preferred[0].direction, Direction::Up);
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

    #[test]
    fn takes_safe_food_even_at_full_health() {
        let me = snake("me", 100, &[(2, 2), (2, 1)]);
        let board = board(
            7,
            7,
            &[(3, 2)],
            &[],
            vec![snake("me", 100, &[(2, 2), (2, 1)])],
        );

        assert_eq!(
            select_direction(&game("standard", None), &board, &me),
            Direction::Right
        );
    }

}
