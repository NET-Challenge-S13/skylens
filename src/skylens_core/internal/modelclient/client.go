// Package modelclient is the core's gRPC client to the model server
// (skylens.model.v1.ModelService, :8101).
//
// Two long-lived loops:
//
//   - Subscriber keeps SubscribeChunks open. Every (re)connect resumes with
//     after_sequence = the last sequence the store holds, so nothing is lost
//     or sent twice across a model or network restart.
//   - Pusher keeps a PushFrames client stream open and feeds it FrameSets.
//     While the model server is down, sets wait in a bounded queue. After a
//     reconnect the most recently sent sets are declared again: a successful
//     Send does not mean the server processed the set, and re-declaring an
//     accepted set is a no-op on the model side (SPEC §14: the core resubmits
//     work when the model server was down).
package modelclient

import (
	"context"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"

	modelv1 "github.com/NET-Challenge-S13/skylens/src/skylens_core/gen/skylens/model/v1"
)

// ChunkStream is the receive side of SubscribeChunks.
type ChunkStream interface {
	Recv() (*modelv1.Chunk, error)
}

// FrameStream is the send side of PushFrames.
type FrameStream interface {
	Send(*modelv1.FrameSet) error
	CloseAndRecv() (*modelv1.PushFramesSummary, error)
}

// API is the slice of ModelService the core uses. The gRPC client satisfies it
// through NewGRPC; tests use fakes.
type API interface {
	SubscribeChunks(ctx context.Context, after uint64) (ChunkStream, error)
	PushFrames(ctx context.Context) (FrameStream, error)
	Health(ctx context.Context) (*modelv1.HealthResponse, error)
}

type grpcAPI struct {
	c modelv1.ModelServiceClient
}

// NewGRPC adapts a generated ModelService client.
func NewGRPC(conn grpc.ClientConnInterface) API {
	return grpcAPI{c: modelv1.NewModelServiceClient(conn)}
}

func (g grpcAPI) SubscribeChunks(ctx context.Context, after uint64) (ChunkStream, error) {
	return g.c.SubscribeChunks(ctx, &modelv1.SubscribeRequest{AfterSequence: after})
}

func (g grpcAPI) PushFrames(ctx context.Context) (FrameStream, error) {
	return g.c.PushFrames(ctx)
}

func (g grpcAPI) Health(ctx context.Context) (*modelv1.HealthResponse, error) {
	return g.c.Health(ctx, &modelv1.HealthRequest{})
}

// Dial opens a lazy client connection to the model server. It does not block;
// the loops below see the server come and go.
func Dial(target string) (*grpc.ClientConn, error) {
	return grpc.NewClient(target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(256<<20), // a refined chunk can be large
			grpc.MaxCallSendMsgSize(64<<20),  // three JPEGs per set
		),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                30 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
	)
}
