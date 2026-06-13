//! `vecvec-proto` — gRPC service/message definitions and the generated tonic/prost
//! stubs.
//!
//! The generated code lives in [`pb`]. [`FILE_DESCRIPTOR_SET`] is the encoded
//! protobuf descriptor used to power server reflection (`grpcurl`, Postman).

/// Generated protobuf messages and tonic service stubs for `vecvec.v1`.
pub mod pb {
    tonic::include_proto!("vecvec.v1");
}

/// Encoded `FileDescriptorSet` for gRPC server reflection.
pub const FILE_DESCRIPTOR_SET: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/vecvec_descriptor.bin"));
