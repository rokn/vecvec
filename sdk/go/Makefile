SRC_PROTO := ../../crates/vecvec-proto/proto/vecvec.proto
DST_PROTO := proto/vecvec/v1/vecvec.proto
GO_PACKAGE := github.com/vecvec/vecvec-go/gen/vecvec/v1;vecvecv1

.PHONY: generate build

# Sync the proto from the upstream crate (re-adding the Go-only go_package
# option) and regenerate the gRPC stubs into gen/.
generate:
	mkdir -p proto/vecvec/v1 gen
	printf 'syntax = "proto3";\n\npackage vecvec.v1;\n\noption go_package = "$(GO_PACKAGE)";\n' > $(DST_PROTO)
	sed '1,3d' $(SRC_PROTO) >> $(DST_PROTO)
	protoc -I proto \
		--go_out=gen --go_opt=paths=source_relative \
		--go-grpc_out=gen --go-grpc_opt=paths=source_relative \
		$(DST_PROTO)

build:
	go build ./...
