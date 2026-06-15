// Package vecvec is a Go SDK for the vecvec vector database.
//
// It wraps the generated gRPC stubs (see the gen/ subpackage) in an ergonomic
// Client that handles dialing, the client-streaming Upsert RPC, and the common
// query/versioning calls.
//
// Quick start:
//
//	ctx := context.Background()
//	c, err := vecvec.Dial(ctx, "127.0.0.1:6334")
//	if err != nil { log.Fatal(err) }
//	defer c.Close()
//
//	c.CreateCollection(ctx, "docs", 3, vecvec.Cosine)
//	ids, _ := c.Upsert(ctx, "docs", []vecvec.Vector{{Values: []float32{0.1, 0.2, 0.3}}})
//	hits, _ := c.Query(ctx, "docs", []float32{0.1, 0.2, 0.3}, 5)
package vecvec

import (
	"context"
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/vecvec/vecvec-go/gen/vecvec/v1"
)

// Metric is the distance metric a collection ranks by.
type Metric = pb.Metric

const (
	Cosine    = pb.Metric_METRIC_COSINE
	Dot       = pb.Metric_METRIC_DOT
	Euclidean = pb.Metric_METRIC_EUCLIDEAN
)

// Vector is a point to upsert: its float values plus an optional JSON payload.
type Vector struct {
	Values  []float32
	Payload string // optional JSON object; empty means none
}

// Hit is one scored search result.
type Hit struct {
	ID    uint64
	Score float32
}

// Client is a connected vecvec client. It is safe for concurrent use.
type Client struct {
	conn        *grpc.ClientConn
	collections pb.CollectionsClient
	points      pb.PointsClient
	query       pb.QueryClient
	versioning  pb.VersioningClient
}

// Dial connects to a vecvec gRPC server (default address 127.0.0.1:6334).
// The connection uses an insecure transport; pass extra grpc.DialOptions
// (e.g. TLS credentials) via opts to override.
func Dial(ctx context.Context, addr string, opts ...grpc.DialOption) (*Client, error) {
	if len(opts) == 0 {
		opts = []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	}
	conn, err := grpc.NewClient(addr, opts...)
	if err != nil {
		return nil, fmt.Errorf("vecvec: dial %s: %w", addr, err)
	}
	return New(conn), nil
}

// New wraps an existing gRPC connection. Use this when you manage the
// connection yourself (custom credentials, interceptors, pooling).
func New(conn *grpc.ClientConn) *Client {
	return &Client{
		conn:        conn,
		collections: pb.NewCollectionsClient(conn),
		points:      pb.NewPointsClient(conn),
		query:       pb.NewQueryClient(conn),
		versioning:  pb.NewVersioningClient(conn),
	}
}

// Close releases the underlying connection. It is a no-op for clients created
// with New from a connection you still own elsewhere.
func (c *Client) Close() error { return c.conn.Close() }

// Raw exposes the generated service clients for RPCs not covered by the
// ergonomic wrapper.
func (c *Client) Raw() (pb.CollectionsClient, pb.PointsClient, pb.QueryClient, pb.VersioningClient) {
	return c.collections, c.points, c.query, c.versioning
}

// ---- Collections ----

// CreateCollection creates a collection with the given dimensionality and metric.
func (c *Client) CreateCollection(ctx context.Context, name string, dim uint32, metric Metric) error {
	_, err := c.collections.Create(ctx, &pb.CreateCollectionRequest{
		Name: name, Dim: dim, Metric: metric,
	})
	return err
}

// ---- Points ----

// Upsert streams vectors into a collection in batches and returns the
// server-assigned ids in input order. batchSize defaults to 1000 if <= 0.
func (c *Client) Upsert(ctx context.Context, collection string, vectors []Vector, batchSize ...int) ([]uint64, error) {
	size := 1000
	if len(batchSize) > 0 && batchSize[0] > 0 {
		size = batchSize[0]
	}
	stream, err := c.points.Upsert(ctx)
	if err != nil {
		return nil, fmt.Errorf("vecvec: open upsert stream: %w", err)
	}
	for start := 0; start < len(vectors); start += size {
		end := min(start+size, len(vectors))
		batch := make([]*pb.Vector, 0, end-start)
		for _, v := range vectors[start:end] {
			pv := &pb.Vector{Values: v.Values}
			if v.Payload != "" {
				p := v.Payload
				pv.Payload = &p
			}
			batch = append(batch, pv)
		}
		if err := stream.Send(&pb.UpsertRequest{Collection: collection, Vectors: batch}); err != nil {
			return nil, fmt.Errorf("vecvec: send upsert batch: %w", err)
		}
	}
	resp, err := stream.CloseAndRecv()
	if err != nil {
		return nil, fmt.Errorf("vecvec: close upsert stream: %w", err)
	}
	return resp.GetIds(), nil
}

// WriteBatch applies deletes + upserts atomically, optionally committing a new
// version afterwards. Pass commit=false and empty message/tag for a plain batch.
func (c *Client) WriteBatch(ctx context.Context, collection string, upserts []Vector, deletes []uint64, commit bool, message, tag string) (*pb.WriteBatchResponse, error) {
	pvs := make([]*pb.Vector, 0, len(upserts))
	for _, v := range upserts {
		pv := &pb.Vector{Values: v.Values}
		if v.Payload != "" {
			p := v.Payload
			pv.Payload = &p
		}
		pvs = append(pvs, pv)
	}
	req := &pb.WriteBatchRequest{
		Collection: collection, Upserts: pvs, Deletes: deletes, Commit: commit,
	}
	if message != "" {
		req.Message = &message
	}
	if tag != "" {
		req.Tag = &tag
	}
	return c.points.WriteBatch(ctx, req)
}

// ---- Query ----

// Query returns the k nearest neighbours of vector in a collection.
func (c *Client) Query(ctx context.Context, collection string, vector []float32, k uint32) ([]Hit, error) {
	resp, err := c.query.Query(ctx, &pb.QueryRequest{
		Collection: collection, Vector: vector, K: k,
	})
	if err != nil {
		return nil, err
	}
	return toHits(resp), nil
}

// QueryAs is Query with a JSON filter and/or time-travel read (version/tag/branch).
// Pass at=nil to read live HEAD and filter="" for no filter.
func (c *Client) QueryAs(ctx context.Context, collection string, vector []float32, k uint32, at *pb.VersionRef, filter string) ([]Hit, error) {
	req := &pb.QueryRequest{Collection: collection, Vector: vector, K: k, At: at}
	if filter != "" {
		req.Filter = &filter
	}
	resp, err := c.query.Query(ctx, req)
	if err != nil {
		return nil, err
	}
	return toHits(resp), nil
}

// Recommend queries by example point ids (positive/negative).
func (c *Client) Recommend(ctx context.Context, collection string, positive, negative []uint64, k uint32, filter string) ([]Hit, error) {
	req := &pb.RecommendRequest{
		Collection: collection, Positive: positive, Negative: negative, K: k,
	}
	if filter != "" {
		req.Filter = &filter
	}
	resp, err := c.query.Recommend(ctx, req)
	if err != nil {
		return nil, err
	}
	return toHits(resp), nil
}

func toHits(resp *pb.QueryResponse) []Hit {
	hits := make([]Hit, 0, len(resp.GetResults()))
	for _, r := range resp.GetResults() {
		hits = append(hits, Hit{ID: r.GetId(), Score: r.GetScore()})
	}
	return hits
}

// ---- Versioning ----

// Commit commits a new version of a collection and returns its version number.
func (c *Client) Commit(ctx context.Context, collection, message, tag string) (uint64, error) {
	req := &pb.CommitRequest{Collection: collection}
	if message != "" {
		req.Message = &message
	}
	if tag != "" {
		req.Tag = &tag
	}
	resp, err := c.versioning.Commit(ctx, req)
	if err != nil {
		return 0, err
	}
	return resp.GetVersion(), nil
}

// ListVersions returns a collection's version history and its current head.
func (c *Client) ListVersions(ctx context.Context, collection string) ([]*pb.VersionInfo, uint64, error) {
	resp, err := c.versioning.ListVersions(ctx, &pb.ListVersionsRequest{Collection: collection})
	if err != nil {
		return nil, 0, err
	}
	return resp.GetVersions(), resp.GetHead(), nil
}

// Diff returns the point ids added and removed between two versions.
func (c *Client) Diff(ctx context.Context, collection string, from, to uint64) (added, removed []uint64, err error) {
	resp, err := c.versioning.Diff(ctx, &pb.DiffRequest{Collection: collection, From: from, To: to})
	if err != nil {
		return nil, nil, err
	}
	return resp.GetAdded(), resp.GetRemoved(), nil
}

// Restore resets a collection's head to a prior version and returns the new version.
func (c *Client) Restore(ctx context.Context, collection string, version uint64) (uint64, error) {
	resp, err := c.versioning.Restore(ctx, &pb.RestoreRequest{Collection: collection, Version: version})
	if err != nil {
		return 0, err
	}
	return resp.GetVersion(), nil
}

// CreateTag tags a version with a name.
func (c *Client) CreateTag(ctx context.Context, collection, name string, version uint64) error {
	_, err := c.versioning.CreateTag(ctx, &pb.TagRequest{Collection: collection, Name: name, Version: version})
	return err
}

// CreateBranch creates a named branch at a version.
func (c *Client) CreateBranch(ctx context.Context, collection, name string, version uint64) error {
	_, err := c.versioning.CreateBranch(ctx, &pb.BranchRequest{Collection: collection, Name: name, Version: version})
	return err
}

// AtVersion / AtTag / AtBranch build a VersionRef for time-travel reads.
func AtVersion(v uint64) *pb.VersionRef { return &pb.VersionRef{Selector: &pb.VersionRef_Version{Version: v}} }
func AtTag(t string) *pb.VersionRef     { return &pb.VersionRef{Selector: &pb.VersionRef_Tag{Tag: t}} }
func AtBranch(b string) *pb.VersionRef  { return &pb.VersionRef{Selector: &pb.VersionRef_Branch{Branch: b}} }
