

bench:
	@cd lsh-rs/bench && cargo +nightly bench

test:
	@cd lsh-rs && cargo test --features stats --lib

python-lib:
	@cd lsh-py && cargo +nightly build --release && cd .. && cp target/release/liblshpy.so lshpy.so

doc:
	@cd lsh-rs && cargo doc --open