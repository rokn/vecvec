PROTO_DIR := ../../crates/vecvec-proto/proto
PROTO := vecvec.proto
GEN_MODULE := github.com/rokn/vecvec/sdk/go/gen

.PHONY: generate build

# Regenerate the gRPC stubs directly from the upstream crate proto, which
# carries the go_package option. No local proto copy is kept. module= mode
# derives the output path from go_package (gen/vecvec/v1/) instead of the
# proto's location, which sits at the crate include-dir root.
generate:
	mkdir -p gen
	protoc -I $(PROTO_DIR) \
		--go_out=gen --go_opt=module=$(GEN_MODULE) \
		--go-grpc_out=gen --go-grpc_opt=module=$(GEN_MODULE) \
		$(PROTO)

build:
	go build ./...
