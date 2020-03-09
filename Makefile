

bench:
	@cd lsh-rs cargo +nightly bench --bin bench --features blas

test:
	@cd lsh-rs && cargo test --lib --features blas