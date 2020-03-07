

bench:
	@cargo +nightly bench --bin bench --features blas

test:
	@cargo test --lib --features blas