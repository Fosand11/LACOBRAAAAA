//! A deliberately conservative Battlesnake policy.
//!
//! The priority order is: avoid a certain death this turn, avoid squares an
//! equal-or-larger head can reach, preserve manoeuvring room, then seek food
//! when health makes it necessary. It is deterministic on purpose: that makes
//! a bad decision reproducible from a game replay and therefore fixable.

use log::info;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::TryFrom;

use crate::{Battlesnake, Board, Coord, Game};

const DIRECTIONS: [Direction; 4] = [
    Direction::Up,
    Direction::Left,
    Direction::Right,
    Direction::Down,
];

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
                y: next.y.rem_euclid(board.height as i32),
            });
        }

        in_bounds(next, board).then_some(next)
    }
}

#[derive(Debug)]
struct Candidate {
    direction: Direction,
    score: i64,
    loses_head_to_head: bool,
}

// info is called when you create your Battlesnake on play.battlesnake.com
// and controls your Battlesnake's appearance.
pub fn info() -> Value {
    info!("INFO");

    json!({
        "apiversion": "1",
        "author": "",
        "color": "#276FBF",
        "head": "safe",
        "tail": "bolt",
    })
}

pub fn start(_game: &Game, _turn: &i32, _board: &Board, _you: &Battlesnake) {
    info!("GAME START");
}

pub fn end(_game: &Game, _turn: &i32, _board: &Board, _you: &Battlesnake) {
    info!("GAME OVER");
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
        return DIRECTIONS
            .iter()
            .copied()
            .find(|direction| {
                direction
                    .next(Point::from(&you.head), board, is_wrapped(game))
                    .is_some()
            })
            .unwrap_or(Direction::Up);
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

    // `>` intentionally preserves DIRECTIONS' stable tie-break order.
    let mut best = pool[0];
    for candidate in pool.into_iter().skip(1) {
        if candidate.score > best.score {
            best = candidate;
        }
    }
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
    let target = direction.next(head, board, wraps)?;
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

    if health_after <= 0 || collides_with_body(target, board, you, grows) {
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
    planning_blocked.remove(&target);

    let (space, distances) = reachable_space(board, target, &planning_blocked, &dangerous, wraps);
    let exits = DIRECTIONS
        .iter()
        .filter_map(|next_direction| next_direction.next(target, board, wraps))
        .filter(|point| !planning_blocked.contains(point) && !dangerous.contains(point))
        .count() as i64;
    let (food_distance, food_is_contested) =
        nearest_food(board, &food, target, my_length, &distances, wraps, &you.id);

    let projected_length = projected_body.len() as i64;
    let space_deficit = (projected_length + 2 - space as i64).max(0);
    let mut score = space as i64 * 140 + exits * 90 + centre_score(target, board);
    score += health_after as i64 * 3;
    score -= hazard_cost as i64 * 120;
    score -= space_deficit * space_deficit * 900;

    if eating {
        // Food is mainly valuable as fuel; growing without a health need is a
        // modest cost because it reduces future manoeuvrability.
        score += if you.health <= 45 {
            (101 - you.health) as i64 * 95
        } else {
            350
        };
    } else if let Some(distance) = food_distance {
        let can_arrive_before_starving = constrictor || distance <= health_after;
        if can_arrive_before_starving && !food_is_contested {
            let urgency = (55 - health_after).max(0) as i64;
            score += urgency * 75 - distance as i64 * 25;
        } else if health_after <= 18 {
            score -= 4_000;
        }
    } else if !constrictor && health_after <= 18 {
        score -= 5_000;
    }

    Some(Candidate {
        direction,
        score,
        loses_head_to_head,
    })
}

fn in_bounds(point: Point, board: &Board) -> bool {
    point.x >= 0 && point.x < board.width && point.y >= 0 && point.y < board.height as i32
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
        next == Some(target) && neck.map_or(true, |point| point != target)
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
                if neck.map_or(true, |neck| neck != point) {
                    territory.insert(point);
                }
            }
        }
    }
    territory
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
            if blocked.contains(&next) || dangerous.contains(&next) || distances.contains_key(&next) {
                continue;
            }
            distances.insert(next, distance + 1);
            queue.push_back(next);
        }
    }

    (distances.len(), distances)
}

fn nearest_food(
    board: &Board,
    food: &HashSet<Point>,
    current_target: Point,
    my_length: i32,
    distances: &HashMap<Point, i32>,
    wraps: bool,
    you_id: &str,
) -> (Option<i32>, bool) {
    let mut closest: Option<(i32, bool)> = None;
    for point in food.iter().copied().filter(|point| *point != current_target) {
        let Some(&distance) = distances.get(&point) else {
            continue;
        };
        // Add the move already selected when comparing race times from the
        // current board state. Manhattan distance intentionally gives opponents
        // the benefit of the doubt and prevents optimistic food races.
        let my_turns = distance + 1;
        let contested = board.snakes.iter().filter(|snake| snake.id != you_id).any(|snake| {
            let opponent_turns = board_distance(Point::from(&snake.head), point, board, wraps);
            snake_length(snake) >= my_length && opponent_turns <= my_turns
        });
        if closest.map_or(true, |(best, _)| distance < best) {
            closest = Some((distance, contested));
        }
    }
    closest.map_or((None, false), |(distance, contested)| {
        (Some(distance), contested)
    })
}

fn board_distance(a: Point, b: Point, board: &Board, wraps: bool) -> i32 {
    let direct_x = (a.x - b.x).abs();
    let direct_y = (a.y - b.y).abs();
    if wraps {
        direct_x.min(board.width - direct_x) + direct_y.min(board.height as i32 - direct_y)
    } else {
        direct_x + direct_y
    }
}

fn centre_score(point: Point, board: &Board) -> i64 {
    // Prefer the centre only as a tie-breaker; usable space dominates this.
    let horizontal = (2 * point.x - (board.width - 1)).abs();
    let vertical = (2 * point.y - (board.height as i32 - 1)).abs();
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
    let chosen = select_direction(game, board, you);
    info!("MOVE {}: {}", turn, chosen.name());
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
        ruleset.insert(
            "settings".to_owned(),
            json!({ "hazardDamagePerTurn": 14 }),
        );
        Game {
            id: "test".to_owned(),
            ruleset,
            map: map.map(str::to_owned),
            timeout: 500,
        }
    }

    fn board(
        width: i32,
        height: u32,
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
    fn response_is_always_a_valid_api_move() {
        let me = snake("me", 100, &[(2, 2), (2, 1)]);
        let board = board(
            5,
            5,
            &[],
            &[],
            vec![snake("me", 100, &[(2, 2), (2, 1)])],
        );
        let response = get_move(&game("standard", None), &7, &board, &me);

        assert!(matches!(
            response["move"].as_str(),
            Some("up" | "down" | "left" | "right")
        ));
    }
}
