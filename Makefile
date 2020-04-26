

bench:
	@cd lsh-rs/bench && cargo +nightly bench

test: clean-lsh-db
	@cd lsh-rs && cargo test --lib

doc:
	@cd lsh-rs && cargo doc --no-deps --open --lib

clean-lsh-db:
	-@cd lsh-rs && rm **.db3