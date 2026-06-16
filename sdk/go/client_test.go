package vecvec_test

import (
	"context"
	"net"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	vecvec "github.com/rokn/vecvec/sdk/go"
)

// serverBinary returns the path to a built vecvec-server, or "" if absent.
func serverBinary() string {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..") // sdk/go -> repo root
	for _, profile := range []string{"debug", "release"} {
		p := filepath.Join(root, "target", profile, "vecvec-server")
		if _, err := exec.LookPath(p); err == nil {
			return p
		}
	}
	return ""
}

// freeAddr asks the OS for an unused localhost TCP port.
func freeAddr(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve port: %v", err)
	}
	defer l.Close()
	return l.Addr().String()
}

// startServer launches vecvec-server on a fresh port + temp data dir and
// returns a client connected to it. It registers cleanup to stop the process.
func startServer(t *testing.T) (*vecvec.Client, context.Context) {
	t.Helper()
	bin := serverBinary()
	if bin == "" {
		t.Skip("vecvec-server not built; run `cargo build -p vecvec-server`")
	}

	grpcAddr := freeAddr(t)
	cmd := exec.Command(bin)
	cmd.Env = append(cmd.Environ(),
		"VECVEC_GRPC_ADDR="+grpcAddr,
		"VECVEC_REST_ADDR="+freeAddr(t),
		"VECVEC_DATA_DIR="+t.TempDir(),
	)
	if err := cmd.Start(); err != nil {
		t.Fatalf("start server: %v", err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})

	ctx := context.Background()
	c, err := vecvec.Dial(ctx, grpcAddr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })

	// Wait for the server to accept RPCs (CreateCollection is idempotent enough
	// for a readiness probe on a unique name).
	deadline := time.Now().Add(15 * time.Second)
	for {
		cctx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
		err := c.CreateCollection(cctx, "_ready", 2, vecvec.Cosine)
		cancel()
		if err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("server not ready: %v", err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	return c, ctx
}

func TestRoundTrip(t *testing.T) {
	c, ctx := startServer(t)

	const coll = "docs"
	if err := c.CreateCollection(ctx, coll, 3, vecvec.Cosine); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	ids, err := c.Upsert(ctx, coll, []vecvec.Vector{
		{Values: []float32{1, 0, 0}, Payload: `{"title":"a"}`},
		{Values: []float32{0, 1, 0}, Payload: `{"title":"b"}`},
		{Values: []float32{0, 0, 1}},
	})
	if err != nil {
		t.Fatalf("upsert: %v", err)
	}
	if len(ids) != 3 {
		t.Fatalf("expected 3 ids, got %d (%v)", len(ids), ids)
	}

	hits, err := c.Query(ctx, coll, []float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected at least one hit")
	}
	// The nearest neighbour of [1,0,0] is the first vector we inserted.
	if hits[0].ID != ids[0] {
		t.Errorf("nearest id = %d, want %d (hits: %+v)", hits[0].ID, ids[0], hits)
	}
}

func TestVersioning(t *testing.T) {
	c, ctx := startServer(t)

	const coll = "versioned"
	if err := c.CreateCollection(ctx, coll, 2, vecvec.Cosine); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	if _, err := c.Upsert(ctx, coll, []vecvec.Vector{{Values: []float32{1, 0}}}); err != nil {
		t.Fatalf("upsert: %v", err)
	}

	version, err := c.Commit(ctx, coll, "first", "v1")
	if err != nil {
		t.Fatalf("commit: %v", err)
	}

	versions, head, err := c.ListVersions(ctx, coll)
	if err != nil {
		t.Fatalf("list versions: %v", err)
	}
	if len(versions) == 0 {
		t.Fatal("expected at least one version")
	}
	if head != version {
		t.Errorf("head = %d, want committed version %d", head, version)
	}
}
