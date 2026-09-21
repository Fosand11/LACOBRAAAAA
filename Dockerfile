FROM rust:1.85-slim

COPY . /usr/app
WORKDIR /usr/app

RUN cargo install --path .

CMD ["starter-snake-rust"]
