FROM rust:1.88-slim

COPY . /usr/app
WORKDIR /usr/app

RUN cargo install --path . --locked

CMD ["starter-snake-rust"]
