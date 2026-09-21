# Battlesnake Rust Starter Project

An official Battlesnake template written in Rust. Get started at [play.battlesnake.com](https://play.battlesnake.com).

![Battlesnake Logo](https://media.battlesnake.com/social/StarterSnakeGitHubRepos_Rust.png)

This project is a great starting point for anyone wanting to program their first Battlesnake in Rust. It can be run locally or easily deployed to a cloud provider of your choosing. See the [Battlesnake API Docs](https://docs.battlesnake.com/api) for more detail. 

[![Run on Replit](https://repl.it/badge/github/BattlesnakeOfficial/starter-snake-rust)](https://replit.com/@Battlesnake/starter-snake-rust)

## Technologies Used

This project uses [Rust](https://www.rust-lang.org/) and [Rocket](https://rocket.rs). It also comes with an optional [Dockerfile](https://docs.docker.com/engine/reference/builder/) to help with deployment.

## Run Your Battlesnake

```sh
cargo run
```

You should see the following output once it is running

```sh
🚀 Rocket has launched from http://0.0.0.0:8000
```

Open [localhost:8000](http://localhost:8000) in your browser and you should see

```json
{"apiversion":"1","author":"","color":"#276FBF","head":"default","tail":"default"}
```

## Strategy and verification

The snake is intentionally survival-first rather than food-first. Every move
is screened for boundaries, bodies, health, hazards, and equal-or-larger
head-to-head collisions. Survivable moves are then ranked by accessible space,
territory won against rival path distances, exits, and safe food races. A
bounded 28-turn self-simulation models tail movement, growth, hunger, and
future enemy arrival zones, avoiding corridors that look open in a one-turn
flood fill but close before the tail can clear them. When it is longer, the
snake also takes only forced head-to-head eliminations: a shorter rival must
have no other legal escape.

Run the deterministic regression suite before deploying:

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo test --release
```

At the default `info` level the server logs only the chosen move and a compact
start/end summary, reducing log overhead under the game timeout. For deep
diagnosis of a local replay, set `RUST_LOG=debug`; that enables rejected move
and candidate-score details.

## Play a Game Locally

Install the [Battlesnake CLI](https://github.com/BattlesnakeOfficial/rules/tree/main/cli)
* You can [download compiled binaries here](https://github.com/BattlesnakeOfficial/rules/releases)
* or [install as a go package](https://github.com/BattlesnakeOfficial/rules/tree/main/cli#installation) (requires Go 1.18 or higher)

Command to run a local game

```sh
battlesnake play -W 11 -H 11 --name 'Rust Starter Project' --url http://localhost:8000 -g solo --browser
```

## Next Steps

Continue with the [Battlesnake Quickstart Guide](https://docs.battlesnake.com/quickstart) to customize and improve your Battlesnake's behavior.

**Note:** To play games on [play.battlesnake.com](https://play.battlesnake.com) you'll need to deploy your Battlesnake to a live web server OR use a port forwarding tool like [ngrok](https://ngrok.com/) to access your server locally.
