// Quickstart for the vecvec Go SDK. Start a server first:
//
//	cargo run -p vecvec-server   # gRPC on 127.0.0.1:6334
//
// then: go run ./example
package main

import (
	"context"
	"fmt"
	"log"

	vecvec "github.com/rokn/vecvec/sdk/go"
)

func main() {
	ctx := context.Background()

	c, err := vecvec.Dial(ctx, "127.0.0.1:6334")
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	const coll = "docs"
	if err := c.CreateCollection(ctx, coll, 3, vecvec.Cosine); err != nil {
		log.Fatalf("create: %v", err)
	}

	ids, err := c.Upsert(ctx, coll, []vecvec.Vector{
		{Values: []float32{0.1, 0.2, 0.3}, Payload: `{"title":"a"}`},
		{Values: []float32{0.9, 0.1, 0.0}, Payload: `{"title":"b"}`},
	})
	if err != nil {
		log.Fatalf("upsert: %v", err)
	}
	fmt.Println("inserted ids:", ids)

	version, err := c.Commit(ctx, coll, "initial load", "v1")
	if err != nil {
		log.Fatalf("commit: %v", err)
	}
	fmt.Println("committed version:", version)

	hits, err := c.Query(ctx, coll, []float32{0.1, 0.2, 0.3}, 5)
	if err != nil {
		log.Fatalf("query: %v", err)
	}
	for _, h := range hits {
		fmt.Printf("id=%d score=%.4f\n", h.ID, h.Score)
	}
}
