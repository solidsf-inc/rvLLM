.PHONY: build check test fmt chat docker validate clean

TOOLCHAIN ?= 1.95.0
CARGO := cargo +$(TOOLCHAIN)
MANIFEST := v3/Cargo.toml

build:
	$(CARGO) build --manifest-path $(MANIFEST) --workspace --release --locked

check:
	$(CARGO) check --manifest-path $(MANIFEST) --workspace --all-targets --locked

test:
	$(CARGO) test --manifest-path $(MANIFEST) --workspace --locked

fmt:
	$(CARGO) fmt --manifest-path $(MANIFEST) --all --check

chat:
	$(CARGO) check --manifest-path chat-client/Cargo.toml --locked

docker:
	scripts/build-docker.sh

validate:
	scripts/validate-local.sh

clean:
	$(CARGO) clean --manifest-path $(MANIFEST)
	$(CARGO) clean --manifest-path chat-client/Cargo.toml
