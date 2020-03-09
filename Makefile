

bench:
	@cd lsh-rs/bench && cargo +nightly bench

test:
	@cd lsh-rs && cargo test --lib