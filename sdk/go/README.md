# vecvec-go

Go SDK for the [vecvec](../..) vector database. It wraps the generated gRPC
stubs in an ergonomic `Client`.

## Install

```sh
go get github.com/rokn/vecvec/sdk/go
```

## Usage

```go
ctx := context.Background()
c, err := vecvec.Dial(ctx, "127.0.0.1:6334") // default gRPC port
if err != nil { log.Fatal(err) }
defer c.Close()

c.CreateCollection(ctx, "docs", 3, vecvec.Cosine)
ids, _ := c.Upsert(ctx, "docs", []vecvec.Vector{
    {Values: []float32{0.1, 0.2, 0.3}, Payload: `{"title":"a"}`},
})
version, _ := c.Commit(ctx, "docs", "initial load", "v1")

hits, _ := c.Query(ctx, "docs", []float32{0.1, 0.2, 0.3}, 5)

// Time-travel read against a tag/version/branch:
hits, _ = c.QueryAs(ctx, "docs", vec, 5, vecvec.AtTag("v1"), "")
```

See [`example/main.go`](example/main.go) for a runnable quickstart.

Anything not covered by the wrapper is reachable via `c.Raw()`, which returns
the four generated service clients.

## Regenerating the stubs

The generated code in `gen/` is checked in. To regenerate after the upstream
`crates/vecvec-proto/proto/vecvec.proto` changes:

```sh
make generate   # copies the proto and runs protoc
```

Requires `protoc`, `protoc-gen-go`, and `protoc-gen-go-grpc` on `PATH`:

```sh
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
```
